use super::{animations::Animate, camera::Camera, canvas::Canvas, world::World};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum AnimationFormat {
    Gif,
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
        self.framerate * self.duration_sec as u32
    }

    fn frame_duration(&self) -> f64 {
        self.duration_sec / self.frame_count() as f64
    }

    fn render_gif(&self, encoder: &mut gif::Encoder<&mut std::fs::File>) {
        let frame_duration = self.frame_duration();
        for i in 0..self.frame_count() {
            let time = i as f64 * frame_duration;
            let canvas = self.render_frame(time);
            encoder
                .write_frame(&gif::Frame::<'_>::from(&canvas))
                .unwrap();
        }
    }

    fn render_to_file(&self, filename: &str, format: AnimationFormat) {
        match format {
            AnimationFormat::Gif => {
                let mut file = std::fs::File::create(filename).unwrap();
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
