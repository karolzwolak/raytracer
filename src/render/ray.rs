pub mod intersection;

use crate::math::{
    matrix::Matrix, point::Point, transform::Transform, tuple::Tuple, vector::Vector,
};

#[derive(Clone, Default)]
pub struct Ray {
    origin: Point,
    direction: Vector,
    /// Precomputed inverse of the direction vector
    /// to avoid division in the intersection calculations
    dir_inv: Vector,
}

impl Transform for Ray {
    fn transform(&mut self, matrix: &Matrix) {
        *self = self.transform_new(matrix);
    }

    fn transform_new(&self, matrix: &Matrix) -> Self {
        Self::new(matrix * self.origin, matrix * self.direction)
    }
}

impl Ray {
    pub fn new(origin: Point, direction: Vector) -> Self {
        let dir_inv = Vector::new(1. / direction.x(), 1. / direction.y(), 1. / direction.z());
        Self {
            origin,
            direction,
            dir_inv,
        }
    }

    pub fn position(&self, time: f64) -> Point {
        self.origin + self.direction * time
    }
    pub fn origin(&self) -> &Point {
        &self.origin
    }
    pub fn direction(&self) -> &Vector {
        &self.direction
    }
    pub fn dir_inv(&self) -> &Vector {
        &self.dir_inv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert_approx_eq_low_prec,
        math::{approx_eq::ApproxEq, tuple::Tuple},
    };

    #[test]
    fn position() {
        let ray = Ray::new(Point::new(2., 3., 4.), Vector::new(1., 0., 0.));

        assert_approx_eq_low_prec!(ray.position(0.), Point::new(2., 3., 4.));
        assert_approx_eq_low_prec!(ray.position(1.), Point::new(3., 3., 4.));
        assert_approx_eq_low_prec!(ray.position(-1.), Point::new(1., 3., 4.));
        assert_approx_eq_low_prec!(ray.position(2.5), Point::new(4.5, 3., 4.));
    }
}
