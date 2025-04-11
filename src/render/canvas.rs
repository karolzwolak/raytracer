use std::{fmt::Display, fs::File, io::Write};

use clap::ValueEnum;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

use crate::math::color::Color;

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum ImageFormat {
    Ppm,
    Png,
}

impl Display for ImageFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageFormat::Ppm => write!(f, "ppm"),
            ImageFormat::Png => write!(f, "png"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Canvas {
    width: usize,
    height: usize,
    pixels: Vec<Color>,
}

impl Canvas {
    pub fn with_color(width: usize, height: usize, color: Color) -> Self {
        Self {
            width,
            height,
            pixels: vec![color; height * width],
        }
    }

    pub fn new(width: usize, height: usize) -> Self {
        Self::with_color(width, height, Color::black())
    }

    fn index(&self, x: usize, y: usize) -> usize {
        self.width * y + x
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn pixel_at(&self, x: usize, y: usize) -> Color {
        self.pixels[self.index(x, y)]
    }
    pub fn write_pixel(&mut self, x: usize, y: usize, new_color: Color) {
        let id = self.index(x, y);
        self.pixels[id] = new_color;
    }

    pub fn pixels_mut(&mut self) -> &mut Vec<Color> {
        &mut self.pixels
    }

    pub fn set_each_pixel<F>(&mut self, fun: F, progressbar: indicatif::ProgressBar)
    where
        F: Fn(usize, usize) -> Color + std::marker::Sync,
    {
        let width = self.width;

        self.pixels
            .par_iter_mut()
            .enumerate()
            .progress_with(progressbar)
            .for_each(|(id, pixel_color)| {
                let x = id % width;
                let y = id / width;
                *pixel_color = fun(x, y);
            })
    }

    pub fn as_u8_rgb(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .flat_map(|color| color.as_scaled_values())
            .collect()
    }
}

impl From<&Canvas> for gif::Frame<'_> {
    fn from(canvas: &Canvas) -> Self {
        gif::Frame::from_rgb(
            canvas.width as u16,
            canvas.height as u16,
            &canvas.as_u8_rgb(),
        )
    }
}

/// saving image in ppm format
impl Canvas {
    const MAX_LINE_LEN: usize = 70;
    fn ppm_header(&self) -> String {
        format!(
            r#"P3
            {} {}
            255
            "#,
            self.width, self.height
        )
    }

    fn ppm_data(&self) -> String {
        let mut line_len = 0;

        self.pixels
            .iter()
            .enumerate()
            .map(|(id, color)| {
                color
                    .as_scaled_values()
                    .into_iter()
                    .enumerate()
                    .map(|(j, val)| {
                        let val_str = val.to_string();
                        let sep = if (id % self.width == 0 && j == 0)
                            || line_len + val_str.len() + 1 > Self::MAX_LINE_LEN
                        {
                            line_len = 0;
                            '\n'
                        } else {
                            ' '
                        };

                        line_len += val_str.len() + 1;
                        if id == 0 && j == 0 {
                            val_str
                        } else if id == self.width * self.height - 1 && j == 2 {
                            format!("{}{}\n", sep, val_str)
                        } else {
                            format!("{}{}", sep, val_str)
                        }
                    })
                    .collect::<String>()
            })
            .collect::<String>()
    }

    pub fn save_to_file(&self, file: File, format: ImageFormat) -> std::io::Result<()> {
        match format {
            ImageFormat::Ppm => self.save_to_ppm(file),
            ImageFormat::Png => self.save_to_png(file),
        }
    }

    pub fn save_to_ppm(&self, mut file: File) -> std::io::Result<()> {
        file.write_all(self.ppm_header().as_bytes())?;
        file.write_all(self.ppm_data().as_bytes())?;
        Ok(())
    }
}

/// savng image in png format
impl Canvas {
    fn colors_to_bytes(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .flat_map(|color| color.as_scaled_values())
            .collect()
    }

    pub fn save_to_png(&self, file: File) -> std::io::Result<()> {
        let mut encoder = png::Encoder::new(file, self.width as u32, self.height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);

        let mut writer = encoder.write_header()?;

        writer
            .write_image_data(&self.colors_to_bytes())
            .map_err(|e| e.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{approx_eq::ApproxEq, assert_approx_eq_low_prec};

    #[test]
    fn index() {
        let width = 5;
        let height = 3;
        let canvas = Canvas::new(width, height);
        assert_eq!(canvas.index(0, 1), width);
        assert_eq!(canvas.index(1, 0), 1);
        assert_eq!(canvas.index(width - 1, height - 1), width * height - 1);
        assert_eq!(canvas.index(1, 2), width * 2 + 1);
        assert_eq!(canvas.index(2, 1), width + 2);
    }

    #[test]
    fn new_blank() {
        let black = Color::black();
        let canvas = Canvas::new(10, 20);
        canvas
            .pixels
            .iter()
            .for_each(|pixel| assert_approx_eq_low_prec!(*pixel, black))
    }

    #[test]
    fn write_pixel() {
        let mut canvas = Canvas::new(10, 10);
        let red = Color::red();

        canvas.write_pixel(2, 3, red);
        assert_approx_eq_low_prec!(canvas.pixel_at(2, 3), red);
    }

    #[test]
    fn ppm_header() {
        let canvas = Canvas::new(5, 3);

        assert_eq!(
            canvas.ppm_header(),
            r#"P3
            5 3
            255
            "#
        );
    }
    #[test]
    fn ppm_pixel_data() {
        let mut canvas = Canvas::new(5, 3);

        canvas.write_pixel(0, 0, Color::new(1.5, 0., 0.));
        canvas.write_pixel(2, 1, Color::new(0., 0.5, 0.));
        canvas.write_pixel(4, 2, Color::new(-1.5, 0., 1.));

        assert_eq!(
            canvas.ppm_data(),
            r#"255 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 128 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 255
"#
        )
    }
    #[test]
    fn split_long_lines_ppm_data() {
        let canvas = Canvas::with_color(10, 2, Color::new(1., 0.8, 0.6));

        assert_eq!(
            canvas.ppm_data(),
            r#"255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204
153 255 204 153 255 204 153 255 204 153 255 204 153
255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204
153 255 204 153 255 204 153 255 204 153 255 204 153
"#
        )
    }
    #[test]
    fn ppm_data_write() -> std::io::Result<()> {
        let mut canvas = Canvas::with_color(10, 10, Color::new(1., 0.8, 0.6));

        canvas.write_pixel(0, 0, Color::new(1.5, 0., 0.));
        canvas.write_pixel(2, 1, Color::new(0., 0.5, 0.));
        canvas.write_pixel(4, 8, Color::new(-1.5, 0., 1.));

        let file = File::create("test.ppm")?;
        canvas.save_to_ppm(file)
    }
    #[test]
    fn ppm_data_ends_with_newline() {
        assert!(Canvas::new(5, 3).ppm_data().ends_with('\n'))
    }
}
