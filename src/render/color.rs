use std::ops;

use crate::approx_eq::ApproxEq;

#[derive(Copy, Clone, Debug)]
pub struct Color {
    r: f64,
    g: f64,
    b: f64,
}

impl Color {
    fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }
    fn black() -> Self {
        Self::new(0., 0., 0.)
    }
    fn red() -> Self {
        Self::new(1., 0., 0.)
    }
    fn green() -> Self {
        Self::new(1., 0., 0.)
    }
    fn blue() -> Self {
        Self::new(1., 0., 0.)
    }
    fn r(&self) -> f64 {
        self.r
    }
    fn b(&self) -> f64 {
        self.b
    }
    fn g(&self) -> f64 {
        self.g
    }
}

impl PartialEq for Color {
    fn eq(&self, other: &Self) -> bool {
        self.r.approq_eq(other.r) && self.g.approq_eq(other.g) && self.b.approq_eq(other.b)
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
