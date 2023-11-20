use std::fs;

use super::color::Color;
use rayon::prelude::*;

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

    pub fn set_each_pixel<F>(&mut self, fun: F)
    where
        F: Fn(usize, usize) -> Color + std::marker::Sync,
    {
        let width = self.width;
        let height = self.height;
        self.pixels
            .par_iter_mut()
            .enumerate()
            .for_each(|(id, pixel_color)| {
                let x = id % width;
                let y = id / height;
                *pixel_color = fun(x, y);
            })
    }
}

// saving to file logic
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
                    .into_iter()
                    .collect::<String>()
            })
            .collect::<String>()
    }

    pub fn save_to_file(&self, file: &str) -> std::io::Result<()> {
        fs::write(file, self.ppm_header() + &self.ppm_data())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            .for_each(|pixel| assert_eq!(*pixel, black))
    }

    #[test]
    fn write_pixel() {
        let mut canvas = Canvas::new(10, 10);
        let red = Color::red();

        canvas.write_pixel(2, 3, red);
        assert_eq!(canvas.pixel_at(2, 3), red);
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

        canvas.save_to_file("test.ppm")
    }
    #[test]
    fn ppm_data_ends_with_newline() {
        assert!(Canvas::new(5, 3).ppm_data().ends_with('\n'))
    }
}
