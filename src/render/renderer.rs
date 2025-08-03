use derive_builder::Builder;

use crate::{
    math::color::Color,
    scene::{Scene, camera::Camera},
    shading::integrator::Integrator,
};

use super::image::Image;

#[derive(PartialEq, Debug, Clone, Builder)]
/// The renderer calculates each pixel color using the integrator and camera.
pub struct Renderer {
    integrator: Integrator,
    camera: Camera,
    /// offset from the center of the pixel
    /// so it should be in range [-0.5, 0.5]
    // TODO: intelligently apply supersampling only if it makes a difference
    #[builder(setter(custom))]
    #[builder(field(
        ty = "Option<usize>",
        build = "Renderer::gen_supersampling_offsets(self.supersampling_offsets.unwrap_or(Renderer::DEFAULT_SUPERSAMPLING_LEVEL))"
    ))]
    supersampling_offsets: Vec<f64>,
    #[builder(default = "false")]
    use_progress_bar: bool,
}

impl RendererBuilder {
    pub fn supersampling_level(&mut self, level: usize) -> &mut Self {
        self.supersampling_offsets = Some(level);
        self
    }
}

impl Renderer {
    pub const MAX_RECURSIVE_DEPTH: usize = 5 - 1;
    pub const DEFAULT_SUPERSAMPLING_LEVEL: usize = 2;

    fn gen_supersampling_offsets(level: usize) -> Vec<f64> {
        match level {
            0 | 1 => vec![0.],
            2 => vec![-0.25, 0.25],
            3 => vec![-0.25, 0., 0.25],
            4 => vec![-0.5, -0.25, 0.25, 0.5],
            _ => vec![-0.5, -0.25, 0., 0.25, 0.5],
        }
    }

    fn color_at_pixel(&self, x: usize, y: usize) -> Color {
        let x = x as f64;
        let y = y as f64;

        let offsets = &self.supersampling_offsets;
        let mut color = Color::black();

        for dx in offsets {
            for dy in offsets {
                color = color
                    + self
                        .integrator
                        .color_at(self.camera.ray_for_pixel(x + dx, y + dy));
            }
        }
        color / offsets.len().pow(2) as f64
    }

    pub fn render_animation_frame(&mut self, progressbar: Option<indicatif::ProgressBar>) -> Image {
        let mut image = self.camera.image();

        image.set_each_pixel(|x: usize, y: usize| self.color_at_pixel(x, y), progressbar);
        image
    }

    pub fn render(&mut self) -> Image {
        let now = std::time::Instant::now();
        self.integrator.scene_mut().compute_bvh();
        println!("partitioning time: {:?}", now.elapsed());

        let mut image = self.camera.image();

        let now = std::time::Instant::now();

        let pb = if self.use_progress_bar {
            let style = indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {wide_bar:.cyan/blue} pixels shaded: {human_pos}/{human_len} {percent}% ({eta})",
        )
        .unwrap();
            let pb = indicatif::ProgressBar::new(image.width() as u64 * image.height() as u64);

            Some(pb.with_style(style))
        } else {
            None
        };
        image.set_each_pixel(|x: usize, y: usize| self.color_at_pixel(x, y), pb);
        println!("render time: {:?}", now.elapsed());
        image
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn integrator(&self) -> &Integrator {
        &self.integrator
    }

    pub fn scene(&self) -> &Scene {
        self.integrator.scene()
    }

    pub fn scene_mut(&mut self) -> &mut Scene {
        self.integrator.scene_mut()
    }

    pub fn use_progress_bar(&self) -> bool {
        self.use_progress_bar
    }
}

#[cfg(test)]
impl Renderer {
    pub fn default_testing(camera: Camera) -> Self {
        RendererBuilder::default()
            .camera(camera)
            .integrator(Integrator::default_testing(Scene::default_testing()))
            .supersampling_level(0)
            .build()
            .unwrap()
    }
}
