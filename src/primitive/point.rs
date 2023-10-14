use super::{tuple::Tuple, vector::Vector};
use crate::approx_eq::ApproxEq;
use std::ops;

#[derive(Copy, Clone, Debug)]
pub struct Point {
    x: f64,
    y: f64,
    z: f64,
}

impl Tuple for Point {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Point { x, y, z }
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
        1.
    }
}

impl Point {
    fn apply_vec(&mut self, vec: Vector) {
        self.x += vec.x();
        self.y += vec.y();
        self.z += vec.z();
    }

    fn zero() -> Self {
        Self {
            x: 0.,
            y: 0.,
            z: 0.,
        }
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.x.approq_eq(other.x) && self.y.approq_eq(other.y) && self.z.approq_eq(other.z)
    }
}
impl ops::Add<Vector> for Point {
    type Output = Point;

    fn add(self, rhs: Vector) -> Self::Output {
        Self {
            x: self.x + rhs.x(),
            y: self.y + rhs.y(),
            z: self.z + rhs.z(),
        }
    }
}

impl ops::Sub for Point {
    type Output = Vector;

    fn sub(self, rhs: Point) -> Self::Output {
        Vector::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl ops::Sub<Vector> for Point {
    type Output = Point;

    fn sub(self, rhs: Vector) -> Self::Output {
        Self {
            x: self.x - rhs.x(),
            y: self.y - rhs.y(),
            z: self.z - rhs.z(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_vec() {
        let mut p1 = Point::new(-2., 3., 1.);
        p1.apply_vec(Vector::new(3., -2., 5.));
        assert_eq!(p1, Point::new(1., 1., 6.));

        let mut p2 = Point::new(3., 2., 1.);
        p2.apply_vec(-Vector::new(5., 6., 7.));
        assert_eq!(p2, Point::new(-2., -4., -6.));
    }

    #[test]
    fn add_vector() {
        assert_eq!(
            Point::new(-2., 3., 1.) + Vector::new(3., -2., 5.),
            Point::new(1., 1., 6.)
        );
    }
    #[test]
    fn sub_vector() {
        assert_eq!(
            Point::new(3., 2., 1.) - Vector::new(5., 6., 7.),
            Point::new(-2., -4., -6.)
        );
    }

    #[test]
    fn sub() {
        assert_eq!(
            Point::new(3., 2., 1.) - Point::new(5., 6., 7.),
            Vector::new(-2., -4., -6.)
        );
    }
}
