use std::ops;

use crate::approx_eq::ApproxEq;

#[derive(Copy, Clone, Debug)]
pub struct Color {
    r: f64,
    g: f64,
    b: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }
    pub fn black() -> Self {
        Self::new(0., 0., 0.)
    }
    pub fn red() -> Self {
        Self::new(1., 0., 0.)
    }
    pub fn green() -> Self {
        Self::new(0., 1., 0.)
    }
    pub fn blue() -> Self {
        Self::new(0., 0., 1.)
    }
    pub fn white() -> Self {
        Self::new(1., 1., 1.)
    }
    pub fn r(&self) -> f64 {
        self.r
    }
    pub fn b(&self) -> f64 {
        self.b
    }
    pub fn g(&self) -> f64 {
        self.g
    }

    fn scale_val_to_u8(v: f64) -> u8 {
        let v = v.clamp(0., 1.);
        (v * 255.).round() as u8
    }

    pub fn as_scaled_values(&self) -> [u8; 3] {
        [
            Self::scale_val_to_u8(self.r),
            Self::scale_val_to_u8(self.g),
            Self::scale_val_to_u8(self.b),
        ]
    }

    pub fn as_ppm_pixel_data(&self) -> String {
        let r = Self::scale_val_to_u8(self.r);
        let g = Self::scale_val_to_u8(self.g);
        let b = Self::scale_val_to_u8(self.b);

        format!("{} {} {}", r, g, b)
    }
}

impl PartialEq for Color {
    fn eq(&self, other: &Self) -> bool {
        self.r.approx_eq(other.r) && self.g.approx_eq(other.g) && self.b.approx_eq(other.b)
    }
}

impl ops::Add for Color {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Color {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}

impl ops::Sub for Color {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Color {
            r: self.r - rhs.r,
            g: self.g - rhs.g,
            b: self.b - rhs.b,
        }
    }
}
impl ops::Mul for Color {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Color {
            r: self.r * rhs.r,
            g: self.g * rhs.g,
            b: self.b * rhs.b,
        }
    }
}
impl ops::Mul<f64> for Color {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Color {
            r: self.r * rhs,
            g: self.g * rhs,
            b: self.b * rhs,
        }
    }
}
impl ops::Div<f64> for Color {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Color {
            r: self.r / rhs,
            g: self.g / rhs,
            b: self.b / rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        assert_eq!(
            Color::new(0.9, 0.6, 0.75) + Color::new(0.7, 0.1, 0.25),
            Color::new(1.6, 0.7, 1.0)
        );
    }

    #[test]
    fn sub() {
        assert_eq!(
            Color::new(0.9, 0.6, 0.75) - Color::new(0.7, 0.1, 0.25),
            Color::new(0.2, 0.5, 0.5)
        );
    }

    #[test]
    fn mul_f64() {
        assert_eq!(Color::new(0.2, 0.3, 0.4) * 2., Color::new(0.4, 0.6, 0.8));
    }
    #[test]
    fn div_f64() {
        assert_eq!(Color::new(0.2, 0.3, 0.4) / 2., Color::new(0.1, 0.15, 0.2));
    }
    #[test]
    fn mul() {
        assert_eq!(
            Color::new(1., 0.2, 0.4) * Color::new(0.9, 1., 0.1),
            Color::new(0.9, 0.2, 0.04)
        );
    }
}
