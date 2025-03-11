use std::{fmt::Display, fs::File};

use clap::ValueEnum;

use super::{animations::Animate, camera::Camera, canvas::Canvas, world::World};

#[derive(Debug, Copy, Clone, PartialEq, ValueEnum)]
pub enum AnimationFormat {
    Gif,
}

impl Display for AnimationFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnimationFormat::Gif => write!(f, "gif"),
        }
    }
}

pub struct Animator {
    world: World,
    camera: Camera,
    framerate: u32,
    duration_sec: f64,
}

impl Animator {
    pub fn new(world: World, camera: Camera, framerate: u32, duration_sec: f64) -> Option<Self> {
        if framerate == 0 || duration_sec == 0. {
            return None;
        }
        Some(Self {
            world,
            camera,
            framerate,
            duration_sec,
        })
    }

    fn render_frame(&self, time: f64) -> Canvas {
        let mut world = self.world.clone();
        world.animate(time);
        world.render(&self.camera)
    }

    fn frame_count(&self) -> u32 {
        (self.framerate as f64 * self.duration_sec).round() as u32
    }

    fn frame_duration(&self) -> f64 {
        1. / self.framerate as f64
    }

    fn render_gif(&self, encoder: &mut gif::Encoder<&mut File>) {
        let frame_duration = self.frame_duration();
        for i in 0..self.frame_count() {
            let time = i as f64 * frame_duration;
            let canvas = self.render_frame(time);
            encoder
                .write_frame(&gif::Frame::<'_>::from(&canvas))
                .unwrap();
        }
    }

    pub fn render_to_file(&self, mut file: File, format: AnimationFormat) {
        match format {
            AnimationFormat::Gif => {
                let mut encoder = gif::Encoder::new(
                    &mut file,
                    self.camera.target_width() as u16,
                    self.camera.target_height() as u16,
                    &[],
                )
                .unwrap();
                self.render_gif(&mut encoder);
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        approx_eq::ApproxEq,
        render::{animator::Animator, camera::Camera, world::World},
    };

    fn animator(framerate: u32, duration_sec: f64) -> Animator {
        let world = World::default_testing();
        let camera = Camera::new(10, 10, 1.);
        Animator::new(world, camera, framerate, duration_sec).unwrap()
    }

    fn default_animator() -> Animator {
        let framerate = 24;
        let duration_sec = 10.0;
        animator(framerate, duration_sec)
    }

    #[test]
    fn frame_count() {
        assert_eq!(default_animator().frame_count(), 240);
        assert_eq!(animator(30, 0.5).frame_count(), 15);
    }

    #[test]
    fn frame_duration() {
        assert!(default_animator().frame_duration().approx_eq(&(1. / 24.)));
    }
}
