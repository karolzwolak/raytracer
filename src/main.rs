use std::{fs::File, path::PathBuf};

use clap::{Args, Parser, Subcommand};
use raytracer::{
    render::{animator::AnimationFormat, image::ImageFormat},
    scene::{camera::Camera, io::yaml, Scene},
};

const DEFAULT_WIDTH: usize = 800;
const DEFAULT_HEIGHT: usize = 800;
const DEFAULT_FOV: f64 = std::f64::consts::FRAC_PI_3;

#[derive(Args, Debug)]
struct AnimationCommand {
    /// The format of the output video.
    #[clap(short = 'f', long, default_value = "mp4")]
    format: AnimationFormat,

    /// The duration of the output video in seconds.
    #[clap(short = 'd', long)]
    duration_sec: f64,

    /// Frames per second of the output video.
    /// Note that not all formats support all framerates.
    /// Use lower framerates when rendering to gif (about 30).
    #[clap(long, default_value = "60")]
    fps: u32,
}

#[derive(Args, Debug)]
struct ImageCommand {
    /// The format of the output image.
    #[clap(short = 'f', long, default_value = "png")]
    format: ImageFormat,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Render a single image.
    Image(ImageCommand),
    /// Render an animation.
    /// Use `animate` field on an object to add animation to it.
    Animate(AnimationCommand),
}

impl Command {
    fn extension(&self) -> String {
        match self {
            Command::Image(image) => image.format.to_string(),
            Command::Animate(animation) => animation.format.to_string(),
        }
    }
}

/// Simple raytracer that renders yaml scenes.
/// Supports basic shapes and materials and .obj models.
/// Can render single images and animations.
#[derive(Parser, Debug)]
#[command(about, long_about = None, disable_help_flag = true)]
struct Cli {
    /// Print help information
    #[arg(long = "help", action = clap::ArgAction::Help)]
    help: Option<bool>,

    #[command(subcommand)]
    command: Command,

    /// The scene file to render.
    scene_file: PathBuf,

    /// The output path of the rendered image.
    /// By default it's `./<scene_filename>.<image_format>`.
    #[clap(short, long)]
    output_path: Option<PathBuf>,

    /// Width (in pixels) of the output image.
    #[clap(short, long, help = format!("Width (in pixels) of the output image.
Overrides the one in the scene file. If not specified anywhere, defaults to {}", DEFAULT_WIDTH))]
    width: Option<usize>,

    /// Height (in pixels) of the output image.
    #[clap(short, long, help = format!("Height (in pixels) of the output image.
Overrides the one in the scene file. If not specified anywhere, defaults to {}", DEFAULT_HEIGHT))]
    height: Option<usize>,

    /// Field of view of the camera in radians.
    /// Overrides the one in the scene file.
    /// If not specified anywhere, defaults to π/3.
    #[clap(long)]
    fov: Option<f64>,

    /// Maximum number of times a ray can bounce (change direction).
    /// Direction change occurs when a ray hits a reflective or refractive surface.
    /// Overrides the one in the scene file.
    #[clap(short, long)]
    depth: Option<usize>,

    /// Controls how many rays are shot per pixel.
    /// In other words, the quality of the anti-aliasing (supersampling).
    /// Overrides the one in the scene file.
    #[clap(short, long)]
    supersampling_level: Option<usize>,
}

fn get_scene_camera(args: &Cli) -> Result<(Scene, Camera), String> {
    let scene_source = std::fs::read_to_string(&args.scene_file)
        .map_err(|e| format!("Failed to read scene file: {}", e))?;
    let (mut scene, camera) =
        yaml::parse_str(&scene_source, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FOV)
            .map_err(|e| format!("Failed to parse scene: {}", e))?;
    if let Some(max_reflective_depth) = args.depth {
        scene.set_max_recursive_depth(max_reflective_depth);
    }
    if let Some(supersampling_level) = args.supersampling_level {
        scene.set_supersampling_level(supersampling_level);
    }
    let camera = Camera::with_inverse_transformation(
        args.width.unwrap_or(camera.target_width()),
        args.height.unwrap_or(camera.target_height()),
        args.fov.unwrap_or(camera.field_of_view()),
        camera.inverse_transformation(),
    );
    Ok((scene, camera))
}

fn render_image(
    image_args: ImageCommand,
    file: File,
    mut scene: Scene,
    camera: Camera,
) -> Result<(), String> {
    let image = scene.render(&camera);
    image
        .save_to_file(file, image_args.format)
        .map_err(|e| format!("Failed to save image: {}", e))
}

fn render_animation(
    animation_args: AnimationCommand,
    file: File,
    scene: Scene,
    camera: Camera,
) -> Result<(), String> {
    let animator = raytracer::render::animator::Animator::new(
        scene,
        camera,
        animation_args.fps,
        animation_args.duration_sec,
    )
    .ok_or_else(|| "Zero framerate or duration".to_string())?;
    animator.render_to_file(file, animation_args.format);

    // animator.render_to_file(&output_path.to_string_lossy(), animation_args.format);
    Ok(())
}

fn render() -> Result<PathBuf, String> {
    let args = Cli::parse();

    let (scene, camera) = get_scene_camera(&args)?;
    let output_path = args.output_path.unwrap_or_else(|| {
        let mut path = args.scene_file.clone();
        path = path.file_name().unwrap().into(); // If scene file is not a file, it would get
                                                 // picked up before parsing
        path.set_extension(args.command.extension());
        path
    });

    let file = std::fs::File::create(&output_path)
        .map_err(|e| format!("Failed to create output file: {}", e))?;

    match args.command {
        Command::Image(image_args) => render_image(image_args, file, scene, camera),
        Command::Animate(animation_args) => render_animation(animation_args, file, scene, camera),
    }?;
    Ok(output_path)
}

fn main() {
    let res = render();
    match res {
        Ok(output_path) => {
            println!("Rendered to {}", output_path.to_string_lossy());
        }
        Err(e) => {
            eprintln!("{}", e);
        }
    }
}
