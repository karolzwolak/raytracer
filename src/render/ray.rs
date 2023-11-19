use crate::primitive::{matrix4::Matrix4, point::Point, vector::Vector};

#[derive(Clone)]
pub struct Ray {
    origin: Point,
    direction: Vector,
}

impl Ray {
    pub fn new(origin: Point, direction: Vector) -> Self {
        Self { origin, direction }
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

    pub fn transform(&self, matrix: Matrix4) -> Self {
        Self::new(matrix * self.origin, matrix * self.direction)
    }
}

#[cfg(test)]
mod tests {
    use crate::primitive::tuple::Tuple;

    use super::*;

    #[test]
    fn position() {
        let ray = Ray::new(Point::new(2., 3., 4.), Vector::new(1., 0., 0.));

        assert_eq!(ray.position(0.), Point::new(2., 3., 4.));
        assert_eq!(ray.position(1.), Point::new(3., 3., 4.));
        assert_eq!(ray.position(-1.), Point::new(1., 3., 4.));
        assert_eq!(ray.position(2.5), Point::new(4.5, 3., 4.));
    }
}
