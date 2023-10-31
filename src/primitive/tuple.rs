use crate::transformation::Transform;

use super::matrix4::Matrix4;

pub trait Tuple {
    fn new(x: f64, y: f64, z: f64) -> Self;

    fn x(&self) -> f64;
    fn y(&self) -> f64;
    fn z(&self) -> f64;
    fn w(&self) -> f64;
}

impl<T> Transform for T
where
    T: Tuple + Copy,
{
    fn transform_borrowed(&mut self, transformation_matrix: &Matrix4) {
        *self = (*transformation_matrix) * (*self);
    }

    fn get_transformed(self) -> Self {
        self
    }
}
