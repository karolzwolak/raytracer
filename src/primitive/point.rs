use super::tuple::Tuple;
use crate::approx_eq::ApproxEq;

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
impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.x.approq_eq(other.x) && self.y.approq_eq(other.y) && self.z.approq_eq(other.z)
    }
}
