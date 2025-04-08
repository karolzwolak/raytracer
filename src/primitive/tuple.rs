use super::matrix::{Matrix, Transform};
use crate::approx_eq::ApproxEq;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn iter() -> impl Iterator<Item = Axis> {
        [Axis::X, Axis::Y, Axis::Z].iter().copied()
    }
}

pub trait Tuple: Sized {
    fn new(x: f64, y: f64, z: f64) -> Self;

    fn x(&self) -> f64;
    fn y(&self) -> f64;
    fn z(&self) -> f64;
    fn w(&self) -> f64;

    fn min(&self, other: &Self) -> Self {
        Self::new(
            other.x().min(self.x()),
            other.y().min(self.y()),
            other.z().min(self.z()),
        )
    }

    fn max(&self, other: &Self) -> Self {
        Self::new(
            other.x().max(self.x()),
            other.y().max(self.y()),
            other.z().max(self.z()),
        )
    }

    fn clamp_min(&self, min: f64) -> Self {
        Self::new(self.x().min(min), self.y().min(min), self.z().min(min))
    }

    fn clamp_max(&self, max: f64) -> Self {
        Self::new(self.x().max(max), self.y().max(max), self.z().max(max))
    }
}

impl<T> ApproxEq for T
where
    T: Tuple,
{
    fn approx_eq_epsilon(&self, other: &Self, epsilon: f64) -> bool {
        self.w() == other.w()
            && self.x().approx_eq_epsilon(&other.x(), epsilon)
            && self.y().approx_eq_epsilon(&other.y(), epsilon)
            && self.z().approx_eq_epsilon(&other.z(), epsilon)
    }
}

impl<T> Transform for T
where
    T: Tuple + Copy,
{
    fn transform(&mut self, matrix: &Matrix) {
        *self = self.transform_new(matrix);
    }

    fn transform_new(&self, matrix: &Matrix) -> Self {
        matrix * (*self)
    }
}
