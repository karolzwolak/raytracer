use std::ops::{self, Index};

use super::{
    point::Point,
    tuple::{Axis, Tuple},
};
use crate::approx_eq::ApproxEq;

#[derive(Copy, Clone, Debug, Default)]
pub struct Vector {
    x: f64,
    y: f64,
    z: f64,
}

impl Tuple for Vector {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vector { x, y, z }
    }

    fn x(&self) -> f64 {
        self.x
    }

    fn y(&self) -> f64 {
        self.y
    }

    fn z(&self) -> f64 {
        self.z
    }

    fn w(&self) -> f64 {
        0.
    }
}

impl Index<Axis> for Vector {
    type Output = f64;

    fn index(&self, index: Axis) -> &Self::Output {
        match index {
            Axis::X => &self.x,
            Axis::Y => &self.y,
            Axis::Z => &self.z,
        }
    }
}

impl Vector {
    pub fn zero() -> Self {
        Self::new(0., 0., 0.)
    }
    pub fn magnitude(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
    pub fn normalize(&self) -> Self {
        let len = self.magnitude();
        if len == 0. {
            return Self::zero();
        }
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    pub fn cross(&self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    pub fn dot(&self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn reflect(&self, normal: Self) -> Self {
        *self - normal * 2. * self.dot(normal)
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        self.approx_eq(other)
    }
}

impl ops::Add<Point> for Vector {
    type Output = Point;

    fn add(self, rhs: Point) -> Self::Output {
        Point::new(self.x + rhs.x(), self.y + rhs.y(), self.z + rhs.z())
    }
}

impl ops::Add for Vector {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl ops::Sub for Vector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl ops::Neg for Vector {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl ops::Mul<f64> for Vector {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: rhs * self.x,
            y: rhs * self.y,
            z: rhs * self.z,
        }
    }
}

impl ops::Div<f64> for Vector {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use super::*;
    use crate::{approx_eq::ApproxEq, assert_approx_eq_low_prec};

    #[test]
    fn add_point() {
        assert_approx_eq_low_prec!(
            Vector::new(3., -2., 5.) + Point::new(-2., 3., 1.),
            Point::new(1., 1., 6.)
        );
    }

    #[test]
    fn add() {
        assert_approx_eq_low_prec!(
            Vector::new(3., -2., 5.) + Vector::new(-2., 3., 1.),
            Vector::new(1., 1., 6.)
        );
    }

    #[test]
    fn sub() {
        assert_approx_eq_low_prec!(
            Vector::new(3., 2., 1.) - Vector::new(5., 6., 7.),
            Vector::new(-2., -4., -6.)
        );
    }

    #[test]
    fn neg() {
        assert_approx_eq_low_prec!(-Vector::new(1., -2., 3.), Vector::new(-1., 2., -3.));
    }

    #[test]
    fn mul_f64() {
        assert_approx_eq_low_prec!(Vector::new(1., -2., 3.) * 3.5, Vector::new(3.5, -7., 10.5));
    }

    #[test]
    fn div_f64() {
        assert_approx_eq_low_prec!(Vector::new(1., -2., 4.) / 2., Vector::new(0.5, -1., 2.));
    }

    #[test]
    fn magnitude() {
        assert_approx_eq_low_prec!(Vector::new(1., 0., 0.).magnitude(), 1.);
        assert_approx_eq_low_prec!(Vector::new(0., 1., 0.).magnitude(), 1.);
        assert_approx_eq_low_prec!(Vector::new(0., 0., 1.).magnitude(), 1.);
        assert_approx_eq_low_prec!(Vector::new(0., 0., 0.).magnitude(), 0.);

        assert_approx_eq_low_prec!(Vector::new(1., 2., 3.).magnitude(), 14_f64.sqrt());
        assert_approx_eq_low_prec!(Vector::new(-1., -2., 3.).magnitude(), 14_f64.sqrt());
    }
    #[test]
    fn normalize() {
        assert_approx_eq_low_prec!(Vector::new(4., 0., 0.).normalize(), Vector::new(1., 0., 0.));
        let sqrt_14 = 14_f64.sqrt();
        assert_approx_eq_low_prec!(
            Vector::new(1., -2., 3.).normalize(),
            Vector::new(1. / sqrt_14, -2. / sqrt_14, 3. / sqrt_14)
        );

        assert_approx_eq_low_prec!(Vector::new(1., 2., 3.).normalize().magnitude(), 1.);
    }

    #[test]
    fn dot_product() {
        assert_approx_eq_low_prec!(Vector::new(1., 2., 3.).dot(Vector::new(2., 3., 4.)), 20.);
    }

    #[test]
    fn cross_product() {
        let v1 = Vector::new(1., 2., 3.);
        let v2 = Vector::new(2., 3., 4.);
        assert_approx_eq_low_prec!(v1.cross(v2), Vector::new(-1., 2., -1.));
        assert_approx_eq_low_prec!(v2.cross(v1), Vector::new(1., -2., 1.));
    }
    #[test]
    fn reflect_vector_approaching_at_45_deg() {
        let v = Vector::new(1., -1., 0.);
        let normal = Vector::new(0., 1., 0.);
        assert_approx_eq_low_prec!(v.reflect(normal), Vector::new(1., 1., 0.));
    }

    #[test]
    fn reflect_vector_off_slanted_surface() {
        let v = Vector::new(0., -1., 0.);
        let normal = Vector::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.);
        assert_approx_eq_low_prec!(v.reflect(normal), Vector::new(1., 0., 0.));
    }
}
