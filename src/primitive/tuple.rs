use super::matrix::{Matrix, Transform};
use crate::approx_eq::ApproxEq;

pub trait Tuple {
    fn new(x: f64, y: f64, z: f64) -> Self;

    fn x(&self) -> f64;
    fn y(&self) -> f64;
    fn z(&self) -> f64;
    fn w(&self) -> f64;
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
