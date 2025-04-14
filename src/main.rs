use std::{fs::File, path::PathBuf};

use clap::{Args, Parser, Subcommand};
use derive_builder::Builder;
use raytracer::{
    render::{
        animator::{AnimationFormat, AnimationRendererBuilder},
        image::ImageFormat,
        renderer::{Renderer, RendererBuilder},
    },
    scene::{
        camera::Camera,
        io::yaml::{self, YamlSceneConfig},
        Scene,
    },
    shading::integrator::IntegratorBuilder,
};

const DEFAULT_WIDTH: usize = 800;
const DEFAULT_HEIGHT: usize = 800;
const DEFAULT_FOV: f64 = std::f64::consts::FRAC_PI_3;
const DEFAULT_FPS: u32 = 60;

#[derive(Args, Debug)]
struct AnimationCommand {
    /// The format of the output video.
    #[clap(short = 'f', long, default_value = "mp4")]
    format: AnimationFormat,

    /// The duration of the output video in seconds.
    #[clap(short = 'd', long)]
    duration_sec: Option<f64>,

    /// Frames per second of the output video.
    /// Note that not all formats support all framerates.
    /// Use lower framerates when rendering to gif (about 30).
    #[clap(long)]
    fps: Option<u32>,
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

#[derive(Debug, PartialEq, Builder)]
struct ImageConfig {
    format: ImageFormat,
}

#[derive(Debug, PartialEq, Builder)]
struct AnimationConfig {
    format: AnimationFormat,
    animation_duration_sec: f64,
    animation_framerate: u32,
}

#[derive(Debug, PartialEq, Builder)]
struct RenderConfig {
    supersampling_level: usize,
    max_reflective_depth: usize,

    camera: Camera,
    scene: Scene,
}

enum ConfigKind {
    Animation(AnimationConfig),
    Image(ImageConfig),
}

struct Config {
    kind: ConfigKind,
    render_config: RenderConfig,
}

impl Config {
    fn merge_from_cli_yaml(cli: Cli, mut yaml: YamlSceneConfig) -> Result<Self, String> {
        let kind = match cli.command {
            Command::Image(image_command) => ConfigKind::Image(
                ImageConfigBuilder::default()
                    .format(image_command.format)
                    .build()
                    .map_err(|e| format!("Failed to build image config: {}", e))?,
            ),
            Command::Animate(animation_command) => {
                let duration = match (animation_command.duration_sec, yaml.animation_duration_sec) {
                    (Some(d), _) => d,
                    (None, Some(d)) => d,
                    (None, None) => {
                        return Err("Animation duration not specified".to_string());
                    }
                };
                let framerate = animation_command
                    .fps
                    .unwrap_or(yaml.animation_framerate.unwrap_or(DEFAULT_FPS));
                let animation_config = AnimationConfigBuilder::default()
                    .format(animation_command.format)
                    .animation_duration_sec(duration)
                    .animation_framerate(framerate)
                    .build()
                    .map_err(|e| format!("Failed to build animation config: {}", e))?;

                ConfigKind::Animation(animation_config)
            }
        };

        yaml.camera_builder.optional_target_width(cli.width);
        yaml.camera_builder.default_target_width(DEFAULT_WIDTH);
        yaml.camera_builder.optional_target_height(cli.height);
        yaml.camera_builder.default_target_height(DEFAULT_HEIGHT);
        yaml.camera_builder.optional_field_of_view(cli.fov);
        yaml.camera_builder.default_field_of_view(DEFAULT_FOV);

        let camera = yaml
            .camera_builder
            .build()
            .map_err(|e| format!("Failed to build camera: {}", e))?;
        let scene = yaml.scene_builder.build();

        let render_config = RenderConfigBuilder::default()
            .supersampling_level(
                cli.supersampling_level.unwrap_or(
                    yaml.supersampling_level
                        .unwrap_or(Renderer::DEFAULT_SUPERSAMPLING_LEVEL),
                ),
            )
            .max_reflective_depth(
                cli.depth.unwrap_or(
                    yaml.max_reflective_depth
                        .unwrap_or(Renderer::MAX_RECURSIVE_DEPTH),
                ),
            )
            .camera(camera)
            .scene(scene)
            .build()
            .map_err(|e| format!("Failed to build render config: {}", e))?;

        Ok(Self {
            kind,
            render_config,
        })
    }

    fn render(self, file: File) -> Result<(), String> {
        let integator = IntegratorBuilder::default()
            .max_recursive_depth(self.render_config.max_reflective_depth)
            .build()
            .map_err(|e| format!("Failed to build integrator: {}", e))?;
        let mut renderer = RendererBuilder::default()
            .supersampling_level(self.render_config.supersampling_level)
            .integrator(integator)
            .camera(self.render_config.camera)
            .build()
            .map_err(|e| format!("Failed to build renderer: {}", e))?;

        match self.kind {
            ConfigKind::Animation(animation_config) => {
                let animation_rendrerer = AnimationRendererBuilder::default()
                    .renderer(renderer)
                    .duration_sec(animation_config.animation_duration_sec)
                    .framerate(animation_config.animation_framerate)
                    .build()
                    .map_err(|e| format!("Failed to build animation renderer: {}", e))?;
                animation_rendrerer.render_to_file(file, animation_config.format);
            }
            ConfigKind::Image(image_config) => {
                let image = renderer.render();
                image
                    .save_to_file(file, image_config.format)
                    .map_err(|e| format!("Failed to save image to file: {}", e))?;
            }
        }
        Ok(())
    }
}

fn parse_yaml_scene(args: &Cli) -> Result<YamlSceneConfig, String> {
    let scene_source = std::fs::read_to_string(&args.scene_file)
        .map_err(|e| format!("Failed to read scene file: {}", e))?;

    yaml::parse_str(&scene_source).map_err(|e| format!("Failed to parse scene: {}", e))
}

fn render() -> Result<PathBuf, String> {
    let args = Cli::parse();

    let output_path = args.output_path.clone().unwrap_or_else(|| {
        let mut path = args.scene_file.clone();
        path = path.file_name().unwrap().into(); // If scene file is not a file, it would get
                                                 // picked up before parsing
        path.set_extension(args.command.extension());
        path
    });
    let yaml_config = parse_yaml_scene(&args)?;

    let config = Config::merge_from_cli_yaml(args, yaml_config)
        .map_err(|e| format!("Failed parse config: {}", e))?;

    let file = std::fs::File::create(&output_path)
        .map_err(|e| format!("Failed to create output file: {}", e))?;
    config.render(file)?;

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
