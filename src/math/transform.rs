use std::ops;

use crate::scene::animation::{Base, SelfInterpolate};

use super::{matrix::Matrix, tuple::Axis};

pub mod local_transform;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transformation {
    Scaling(f64, f64, f64),
    Translation(f64, f64, f64),
    Rotation(Axis, f64),
    Shearing(f64, f64, f64, f64, f64, f64),
    Identity,
}

impl Transformation {
    pub fn scaling_uniform(f: f64) -> Self {
        Self::Scaling(f, f, f)
    }
}

impl From<Transformation> for Matrix {
    fn from(val: Transformation) -> Self {
        match val {
            Transformation::Scaling(x, y, z) => Matrix::scaling(x, y, z),
            Transformation::Translation(x, y, z) => Matrix::translation(x, y, z),
            Transformation::Rotation(axis, radians) => match axis {
                Axis::X => Matrix::rotation_x(radians),
                Axis::Y => Matrix::rotation_y(radians),
                Axis::Z => Matrix::rotation_z(radians),
            },
            Transformation::Shearing(xpy, xpz, ypx, ypz, zpx, zpy) => {
                Matrix::shearing(xpy, xpz, ypx, ypz, zpx, zpy)
            }
            Transformation::Identity => Matrix::identity(),
        }
    }
}

impl Base for Transformation {
    fn base(&self) -> Self {
        match self {
            Self::Scaling(_, _, _) => Self::Scaling(1., 1., 1.),
            Self::Translation(_, _, _) => Self::Translation(0., 0., 0.),
            Self::Rotation(axis, _) => Self::Rotation(*axis, 0.),
            Self::Shearing(_, _, _, _, _, _) => Self::Shearing(0., 0., 0., 0., 0., 0.),
            Self::Identity => Self::Identity,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transformations {
    data: Vec<Transformation>,
}

impl Transformations {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    pub fn with_vec(vec: Vec<Transformation>) -> Self {
        Self { data: vec }
    }
    pub fn vec(&self) -> &[Transformation] {
        &self.data
    }
    pub fn push(&mut self, t: Transformation) {
        self.data.push(t);
    }
    pub fn extend(&mut self, other: &Self) {
        self.data.extend(other.data.iter().copied());
    }
}

impl SelfInterpolate for Transformations {
    fn interpolated(&self, at: f64) -> Self {
        Self::with_vec(self.data.interpolated(at))
    }
}

impl Default for Transformations {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Vec<Transformation>> for Transformations {
    fn from(val: Vec<Transformation>) -> Self {
        Self { data: val }
    }
}

impl From<&[Transformation]> for Transformations {
    fn from(val: &[Transformation]) -> Self {
        Self { data: val.to_vec() }
    }
}
impl From<&Transformations> for Matrix {
    fn from(val: &Transformations) -> Self {
        Matrix::from(val.vec())
    }
}

impl From<Transformations> for Matrix {
    fn from(val: Transformations) -> Self {
        Matrix::from(val.vec())
    }
}

impl From<Vec<Transformation>> for Matrix {
    fn from(val: Vec<Transformation>) -> Self {
        Matrix::from(&val[..])
    }
}

impl From<&[Transformation]> for Matrix {
    fn from(val: &[Transformation]) -> Self {
        val.iter().fold(Matrix::identity(), |acc, t| {
            acc.transform_new(&Matrix::from(*t))
        })
    }
}

impl From<Transformations> for Vec<Transformation> {
    fn from(val: Transformations) -> Self {
        val.data
    }
}

impl ops::Mul<f64> for Transformation {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        match self {
            Self::Scaling(fx, fy, fz) => Self::Scaling(fx * rhs, fy * rhs, fz * rhs),
            Self::Translation(x, y, z) => Self::Translation(x * rhs, y * rhs, z * rhs),
            Self::Rotation(axis, radians) => Self::Rotation(axis, radians * rhs),
            Self::Shearing(xpy, xpz, ypx, ypz, zpx, zpy) => Self::Shearing(
                xpy * rhs,
                xpz * rhs,
                ypx * rhs,
                ypz * rhs,
                zpx * rhs,
                zpy * rhs,
            ),
            Self::Identity => Self::Identity,
        }
    }
}

impl ops::Add<Transformation> for Transformation {
    type Output = Transformation;

    fn add(self, rhs: Transformation) -> Self::Output {
        match (self, rhs) {
            (Self::Scaling(x1, y1, z1), Self::Scaling(x2, y2, z2)) => {
                Self::Scaling(x1 + x2, y1 + y2, z1 + z2)
            }
            (Self::Translation(x1, y1, z1), Self::Translation(x2, y2, z2)) => {
                Self::Translation(x1 + x2, y1 + y2, z1 + z2)
            }
            (Self::Rotation(axis1, radians1), Self::Rotation(axis2, radians2)) => {
                assert_eq!(axis1, axis2);
                Self::Rotation(axis1, radians1 + radians2)
            }
            (
                Self::Shearing(xpy1, xpz1, ypx1, ypz1, zpx1, zpy1),
                Self::Shearing(xpy2, xpz2, ypx2, ypz2, zpx2, zpy2),
            ) => Self::Shearing(
                xpy1 + xpy2,
                xpz1 + xpz2,
                ypx1 + ypx2,
                ypz1 + ypz2,
                zpx1 + zpx2,
                zpy1 + zpy2,
            ),
            (Self::Identity, Self::Identity) => Self::Identity,
            _ => panic!("Cannot add different transformations"),
        }
    }
}

impl ops::Sub<Transformation> for Transformation {
    type Output = Self;

    fn sub(self, rhs: Transformation) -> Self::Output {
        self + (rhs * -1.)
    }
}

pub trait Transform: Sized + Clone {
    fn transform(&mut self, matrix: &Matrix);

    fn transform_new(&self, matrix: &Matrix) -> Self {
        let mut copy = self.clone();
        copy.transform(matrix);
        copy
    }

    fn transformed(self) -> Self {
        self
    }

    fn transform_chain(&mut self, transformation: &Matrix) -> &mut Self {
        self.transform(transformation);
        self
    }

    fn translate(&mut self, x: f64, y: f64, z: f64) -> &mut Self {
        self.transform_chain(&Matrix::translation(x, y, z))
    }

    fn scale(&mut self, x: f64, y: f64, z: f64) -> &mut Self {
        self.transform_chain(&Matrix::scaling(x, y, z))
    }

    fn scale_uniform(&mut self, factor: f64) -> &mut Self {
        self.transform_chain(&Matrix::scaling_uniform(factor))
    }

    fn rotate_x(&mut self, radians: f64) -> &mut Self {
        self.transform_chain(&Matrix::rotation_x(radians))
    }

    fn rotate_y(&mut self, radians: f64) -> &mut Self {
        self.transform_chain(&Matrix::rotation_y(radians))
    }

    fn rotate_z(&mut self, radians: f64) -> &mut Self {
        self.transform_chain(&Matrix::rotation_z(radians))
    }

    fn sheare(
        &mut self,
        x_prop_y: f64,
        x_prop_z: f64,
        y_prop_x: f64,
        y_prop_z: f64,
        z_prop_x: f64,
        z_prop_y: f64,
    ) -> &mut Self {
        self.transform_chain(&Matrix::shearing(
            x_prop_y, x_prop_z, y_prop_x, y_prop_z, z_prop_x, z_prop_y,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::math::approx_eq::ApproxEq;
    use std::f64::consts;

    use crate::{
        assert_approx_eq_low_prec,
        math::{
            matrix::Matrix,
            transform::{Transform, Transformation, Transformations},
            tuple::Axis,
        },
        scene::animation::{Base, SelfInterpolate},
    };

    #[test]
    fn default_interpolate() {
        let transforms = Transformations::from(vec![
            Transformation::Scaling(1., 2., 3.).base(),
            Transformation::Translation(4., 5., 6.).base(),
            Transformation::Rotation(Axis::X, consts::FRAC_PI_2).base(),
            Transformation::Shearing(1., 2., 3., 4., 5., 6.).base(),
        ]);
        let factor = 0.25;
        let interpolated = transforms.interpolated(factor);

        assert_eq!(transforms, interpolated);
    }

    #[test]
    fn interpolate_transform() {
        let transforms = Transformations::from(vec![
            Transformation::Scaling(1., 2., 3.),
            Transformation::Translation(4., 5., 6.),
            Transformation::Rotation(Axis::X, consts::FRAC_PI_2),
            Transformation::Shearing(1., 2., 3., 4., 5., 6.),
        ]);
        let factor = 0.25;
        let expected = Transformations::from(vec![
            Transformation::Scaling(1., 1.25, 1.5),
            Transformation::Translation(1., 1.25, 1.5),
            Transformation::Rotation(Axis::X, consts::FRAC_PI_2 * 0.25),
            Transformation::Shearing(0.25, 0.5, 0.75, 1., 1.25, 1.5),
        ]);
        let interpolated = transforms.interpolated(factor);

        assert_eq!(interpolated, expected);
    }

    #[test]
    fn transformation_vec_to_matrix() {
        let transformations = Transformations::from(vec![
            Transformation::Scaling(2., -3.5, 4.),
            Transformation::Rotation(Axis::X, consts::FRAC_PI_2),
            Transformation::Shearing(0., 1., 2., -3., 4., 5.),
            Transformation::Translation(5., 6., 7.),
            Transformation::Rotation(Axis::Y, consts::FRAC_PI_4),
            Transformation::Rotation(Axis::Z, -consts::FRAC_PI_6),
        ]);
        let expected = Matrix::scaling(2., -3.5, 4.)
            .rotate_x(consts::FRAC_PI_2)
            .sheare(0., 1., 2., -3., 4., 5.)
            .translate(5., 6., 7.)
            .rotate_y(consts::FRAC_PI_4)
            .rotate_z(-consts::FRAC_PI_6)
            .transformed();

        assert_approx_eq_low_prec!(Matrix::from(&transformations), expected);
    }
}
