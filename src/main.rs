use std::path::PathBuf;

use clap::Parser;
use raytracer::{
    render::{camera::Camera, canvas::ImageFormat},
    yaml,
};

const DEFAULT_WIDTH: usize = 800;
const DEFAULT_HEIGHT: usize = 800;
const DEFAULT_FOV: f64 = std::f64::consts::FRAC_PI_3;

/// Simple raytracer renderer
/// Renders scenes from yaml files
/// Supports basic shapes and materials and .obj models
#[derive(Parser, Debug)]
#[command(about, long_about = None)]
struct Args {
    /// The scene file to render
    scene_file: PathBuf,

    /// The format of the output image
    #[clap(short = 'f', long, default_value = "png")]
    image_format: ImageFormat,

    /// The output path of the rendered image.
    /// By default it's `./<scene_filename>.<image_format>`
    #[clap(short, long)]
    output_path: Option<PathBuf>,

    /// Width (in pixels) of the output image.
    #[clap(long, help = format!("Width (in pixels) of the output image.
Overrides the one in the scene file. If not specified anywhere, defaults to {}", DEFAULT_WIDTH))]
    width: Option<usize>,

    /// Height (in pixels) of the output image.
    #[clap(long, help = format!("Height (in pixels) of the output image.
Overrides the one in the scene file. If not specified anywhere, defaults to {}", DEFAULT_HEIGHT))]
    #[clap(long)]
    height: Option<usize>,

    /// Field of view of the camera in radians.
    /// Overrides the one in the scene file
    /// If not specified anywhere, defaults to π/3
    #[clap(long)]
    fov: Option<f64>,

    /// Maximum number of times a ray can bounce off a reflective surface.
    /// Overrides the one in the scene file
    #[clap(short, long)]
    max_reflective_depth: Option<usize>,

    /// Controls how many rays are shot per pixel.
    /// In other words, the quality of the anti-aliasing (supersampling).
    /// Overrides the one in the scene file
    #[clap(short, long)]
    supersampling_level: Option<usize>,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let scene_source = std::fs::read_to_string(&args.scene_file)
        .map_err(|e| format!("Failed to read scene file: {}", e))?;
    let (mut world, camera) =
        yaml::parse_str(&scene_source, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FOV)
            .map_err(|e| format!("Failed to parse scene: {}", e))?;
    if let Some(max_reflective_depth) = args.max_reflective_depth {
        world.set_max_recursive_depth(max_reflective_depth);
    }
    if let Some(supersampling_level) = args.supersampling_level {
        world.set_supersampling_level(supersampling_level);
    }
    let camera = Camera::with_inverse_transformation(
        args.width.unwrap_or(camera.target_width()),
        args.height.unwrap_or(camera.target_height()),
        args.fov.unwrap_or(camera.field_of_view()),
        camera.inverse_transformation(),
    );
    let canvas = world.render(&camera);
    let output_path = args.output_path.unwrap_or_else(|| {
        let mut path = args.scene_file.clone();
        path = path.file_name().unwrap().into(); // If scene file is not a file, it would get
                                                 // picked up before parsing
        path.set_extension(args.image_format.to_string());
        path
    });
    canvas
        .save_to_file(&output_path, args.image_format)
        .map_err(|e| format!("Failed to save image: {}", e))?;
    println!("Image saved to {:?}", output_path);
    Ok(())
}
