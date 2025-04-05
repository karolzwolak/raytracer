use std::{fmt::Display, fs::File, io::Write, ops::Deref};

use clap::ValueEnum;
use indicatif::ProgressIterator;
use minimp4::Mp4Muxer;
use openh264::{
    encoder::Encoder,
    formats::{RgbSliceU8, YUVBuffer},
};
use std::thread;
use std::time::Duration;
use webp::WebPConfig;

use super::{camera::Camera, canvas::Canvas, world::World};

#[derive(Debug, Copy, Clone, PartialEq, ValueEnum)]
pub enum AnimationFormat {
    Gif,
    Mp4,
    Webp,
}

impl Display for AnimationFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnimationFormat::Gif => write!(f, "gif"),
            AnimationFormat::Mp4 => write!(f, "mp4"),
            AnimationFormat::Webp => write!(f, "webp"),
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

    fn render_frame(&self, time: f64, progressbar: indicatif::ProgressBar) -> Canvas {
        let mut world = self.world.clone();
        world.animate(time);
        world.render_animation_frame(&self.camera, progressbar)
    }

    fn frame_count(&self) -> u32 {
        (self.framerate as f64 * self.duration_sec).round() as u32
    }

    fn frame_duration(&self) -> f64 {
        1. / self.framerate as f64
    }

    fn render_animation<F>(&self, mut encode_fun: F)
    where
        F: FnMut(Canvas),
    {
        let main_bar = indicatif::ProgressBar::new(self.frame_count() as u64).with_style(indicatif::ProgressStyle::with_template(
            "[{elapsed_precise}] {wide_bar:.cyan/blue} rendering frame: {human_pos}/{human_len} {percent}% ({eta})",
        ).unwrap());

        let pixels_count = self.camera.target_width() as u64 * self.camera.target_height() as u64;
        let frame_bar = indicatif::ProgressBar::new(pixels_count).with_style(indicatif::ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:.cyan/blue} pixels shaded: {human_pos}/{human_len} {percent}% ({eta})",
        )
        .unwrap()
            .progress_chars("=>-")
        );

        let multi_bar = indicatif::MultiProgress::new();
        let main_bar = multi_bar.add(main_bar);

        // Spawn a thread to update the main progress bar's elapsed time
        let main_bar_clone = main_bar.clone();
        thread::spawn(move || loop {
            thread::sleep(Duration::from_millis(100));
            main_bar_clone.tick();
        });

        let frame_duration = self.frame_duration();
        (0..self.frame_count())
            .progress_with(main_bar)
            .for_each(|i| {
                let time = i as f64 * frame_duration;
                let bar = multi_bar.add(frame_bar.clone());
                let canvas = self.render_frame(time, bar.clone());
                bar.reset();
                encode_fun(canvas);
            });
    }

    fn render_gif(&self, encoder: &mut gif::Encoder<&mut File>) {
        self.render_animation(|canvas| {
            encoder
                .write_frame(&gif::Frame::<'_>::from(&canvas))
                .unwrap();
        });
    }

    fn render_mp4(&self, file: File) {
        let mut muxer = Mp4Muxer::new(file);
        let width = self.camera.target_width();
        let height = self.camera.target_height();

        muxer.init_video(width as i32, height as i32, false, "title");
        let mut h264_encoder = Encoder::new().unwrap();

        self.render_animation(|canvas| {
            let bytes = canvas.as_u8_rgb();

            let rgb_source = RgbSliceU8::new(&bytes, (width, height));
            let buf = YUVBuffer::from_rgb_source(rgb_source);
            let bitstream = h264_encoder.encode(&buf).unwrap();

            muxer.write_video_with_fps(&bitstream.to_vec(), self.framerate);
        });
        muxer.close();
    }

    fn render_webp(&self, mut file: File) {
        let width = self.camera.target_width() as u32;
        let height = self.camera.target_height() as u32;
        let mut config =
            WebPConfig::new_with_preset(libwebp_sys::WebPPreset::WEBP_PRESET_PICTURE, 95.).unwrap();
        config.method = 6;
        config.segments = 1;
        config.filter_strength = 10;
        config.autofilter = 0;
        let mut encoder = webp::AnimEncoder::new(width, height, &config);

        let mut frame_buffer = Vec::with_capacity(self.frame_count() as usize);
        self.render_animation(|canvas| {
            let bytes = canvas.as_u8_rgb();
            frame_buffer.push(bytes);
        });

        let frametime_ms = 1000 / self.framerate as usize;
        frame_buffer.iter().enumerate().for_each(|(id, bytes)| {
            let timestamp_ms = id * frametime_ms;
            let frame = webp::AnimFrame::from_rgb(bytes, width, height, timestamp_ms as i32);
            encoder.add_frame(frame);
        });
        println!("Encoding webp...");
        let res = encoder.encode();
        file.write_all(res.deref()).unwrap();
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
                encoder.set_repeat(gif::Repeat::Infinite).unwrap();
                self.render_gif(&mut encoder);
            }
            AnimationFormat::Mp4 => self.render_mp4(file),
            AnimationFormat::Webp => self.render_webp(file),
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
