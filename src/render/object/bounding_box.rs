use crate::{
    approx_eq::{self, ApproxEq},
    primitive::{
        matrix::{Matrix, Transform},
        point::Point,
        tuple::Tuple,
    },
    render::{
        color::Color,
        material::Material,
        object::{shape::Shape, Object},
        pattern::Pattern,
        ray::Ray,
    },
};

#[derive(Clone, Debug)]
/// Axis-aligned bounding box
pub struct BoundingBox {
    pub min: Point,
    pub max: Point,
}

impl Transform for BoundingBox {
    fn transform(&mut self, matrix: &Matrix) {
        *self = self.transform_new(matrix);
    }

    fn transform_new(&self, matrix: &Matrix) -> Self {
        let mut new_bounds = BoundingBox::empty();
        let corners = vec![
            self.min,
            Point::new(self.min.x(), self.min.y(), self.max.z()),
            Point::new(self.min.x(), self.max.y(), self.min.z()),
            Point::new(self.min.x(), self.max.y(), self.max.z()),
            Point::new(self.max.x(), self.min.y(), self.min.z()),
            Point::new(self.max.x(), self.min.y(), self.max.z()),
            Point::new(self.max.x(), self.max.y(), self.min.z()),
            self.max,
        ];
        for corner in corners {
            new_bounds.add_point(matrix * corner);
        }
        new_bounds
    }
}

impl BoundingBox {
    pub fn empty() -> Self {
        Self {
            min: Point::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Point::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.min.x() > self.max.x() && self.min.y() > self.max.y() && self.min.z() > self.max.z()
    }
    pub fn add_point(&mut self, point: Point) {
        self.min = Point::new(
            self.min.x().min(point.x()),
            self.min.y().min(point.y()),
            self.min.z().min(point.z()),
        );
        self.max = Point::new(
            self.max.x().max(point.x()),
            self.max.y().max(point.y()),
            self.max.z().max(point.z()),
        );
    }
    pub fn add_bounding_box(&mut self, other: &BoundingBox) {
        if other.is_empty() {
            return;
        }
        self.add_point(other.min);
        self.add_point(other.max);
    }
    fn axis_intersection_times(&self, origin: f64, dir_inv: f64, min: f64, max: f64) -> (f64, f64) {
        let tmin_numerator = min - origin;
        let tmax_numerator = max - origin;

        let tmin = tmin_numerator * dir_inv;
        let tmax = tmax_numerator * dir_inv;

        if tmin < tmax {
            (tmin, tmax)
        } else {
            (tmax, tmin)
        }
    }
    pub fn is_intersected(&self, ray: &Ray) -> bool {
        let (xtmin, xtmax) = self.axis_intersection_times(
            ray.origin().x(),
            ray.dir_inv().x(),
            self.min.x(),
            self.max.x(),
        );
        let (ytmin, ytmax) = self.axis_intersection_times(
            ray.origin().y(),
            ray.dir_inv().y(),
            self.min.y(),
            self.max.y(),
        );
        let tmin = xtmin.max(ytmin);
        let tmax = xtmax.min(ytmax);

        if tmin > tmax {
            return false;
        }
        let (ztmin, ztmax) = self.axis_intersection_times(
            ray.origin().z(),
            ray.dir_inv().z(),
            self.min.z(),
            self.max.z(),
        );

        let tmin = tmin.max(ztmin);
        let tmax = tmax.min(ztmax);

        tmin <= tmax
    }
    pub fn split_along_longest_axis(&self) -> (BoundingBox, BoundingBox) {
        let x_len = self.max.x() - self.min.x();
        let y_len = self.max.y() - self.min.y();
        let z_len = self.max.z() - self.min.z();

        let longest_axis = x_len.max(y_len).max(z_len);

        let (mut x0, mut y0, mut z0) = (self.min.x(), self.min.y(), self.min.z());
        let (mut x1, mut y1, mut z1) = (self.max.x(), self.max.y(), self.max.z());

        if longest_axis.approx_eq(&x_len) {
            x0 += x_len / 2.;
            x1 = x0;
        } else if longest_axis.approx_eq(&y_len) {
            y0 += y_len / 2.;
            y1 = y0;
        } else {
            z0 += z_len / 2.;
            z1 = z0;
        }
        let mid_min = Point::new(x0, y0, z0);
        let mid_max = Point::new(x1, y1, z1);
        (
            BoundingBox {
                min: self.min,
                max: mid_max,
            },
            BoundingBox {
                min: mid_min,
                max: self.max,
            },
        )
    }
    pub fn is_infinitely_large(&self) -> bool {
        let len = self.max - self.min;
        len.magnitude() == f64::INFINITY
    }
    pub fn split_n(&self, n: usize) -> Vec<BoundingBox> {
        let (a, b) = self.split_along_longest_axis();
        let mut result = vec![a, b];
        for _ in 1..n {
            for bb in std::mem::take(&mut result) {
                let (left, right) = bb.split_along_longest_axis();
                result.push(left);
                result.push(right);
            }
        }
        result
    }

    pub fn as_object(&self) -> Object {
        // render slightly bigger box to avoid z-fighting
        const LEN_FACTOR: f64 = 0.5 * (1. + approx_eq::LOW_PREC_EPSILON);

        let x_len = self.max.x() - self.min.x();
        let y_len = self.max.y() - self.min.y();
        let z_len = self.max.z() - self.min.z();
        let center = self.center();
        let pattern = Pattern::Const(Color::new(0.5, 0.5, 0.5));
        Object::primitive(
            Shape::Cube,
            Material {
                pattern,
                transparency: 1.,
                reflectivity: 0.,
                ambient: 0.1,
                ..Material::air()
            },
            Matrix::identity()
                .scale(x_len * LEN_FACTOR, y_len * LEN_FACTOR, z_len * LEN_FACTOR)
                .translate(center.x(), center.y(), center.z())
                .transformed(),
        )
    }
    pub fn center(&self) -> Point {
        Point::new(
            (self.min.x() + self.max.x()) / 2.,
            (self.min.y() + self.max.y()) / 2.,
            (self.min.z() + self.max.z()) / 2.,
        )
    }
    pub fn contains_point(&self, point: &Point) -> bool {
        (self.min.x() < point.x() || self.min.x().approx_eq(&point.x()))
            && (self.min.y() < point.y() || self.min.y().approx_eq(&point.y()))
            && (self.min.z() < point.z() || self.min.z().approx_eq(&point.z()))
            && (self.max.x() > point.x() || self.max.x().approx_eq(&point.x()))
            && (self.max.y() > point.y() || self.max.y().approx_eq(&point.y()))
            && (self.max.z() > point.z() || self.max.z().approx_eq(&point.z()))
    }
    pub fn contains_other(&self, other: &BoundingBox) -> bool {
        self.contains_point(&other.min) && self.contains_point(&other.max)
    }
    pub fn distance(&self, other: &BoundingBox) -> f64 {
        (self.center() - other.center()).magnitude()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::matrix::Matrix;

    #[test]
    fn test_empty() {
        let bb = BoundingBox::empty();
        assert_eq!(
            bb.min,
            Point::new(f64::INFINITY, f64::INFINITY, f64::INFINITY)
        );
        assert_eq!(
            bb.max,
            Point::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY)
        );
    }

    #[test]
    fn test_is_empty() {
        let mut bb = BoundingBox::empty();
        assert!(bb.is_empty());
        bb.add_point(Point::zero());
        println!("{:?}", bb);
        assert!(!bb.is_empty());
    }

    #[test]
    fn test_add_point() {
        let mut bb = BoundingBox::empty();
        bb.add_point(Point::new(1.0, 2.0, 3.0));
        assert_eq!(bb.min, Point::new(1.0, 2.0, 3.0));
        assert_eq!(bb.max, Point::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_add_bounding_box() {
        let mut bb1 = BoundingBox::empty();
        bb1.add_point(Point::new(1.0, 2.0, 3.0));
        let mut bb2 = BoundingBox::empty();
        bb2.add_point(Point::new(4.0, 5.0, 6.0));
        bb1.add_bounding_box(&bb2);
        assert_eq!(bb1.min, Point::new(1.0, 2.0, 3.0));
        assert_eq!(bb1.max, Point::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn test_contains_point() {
        let mut bb = BoundingBox::empty();
        bb.add_point(Point::new(1.0, 2.0, 3.0));
        bb.add_point(Point::new(4.0, 5.0, 6.0));
        assert!(bb.contains_point(&Point::new(2.0, 3.0, 4.0)));
        assert!(!bb.contains_point(&Point::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn test_transformed() {
        let mut bb = BoundingBox::empty();
        bb.add_point(Point::new(1.0, 2.0, 3.0));
        bb.add_point(Point::new(4.0, 5.0, 6.0));

        let transform_matrix = Matrix::translation(1.0, 1.0, 1.0);
        bb.transform(&transform_matrix);

        assert_eq!(bb.min, Point::new(2.0, 3.0, 4.0));
        assert_eq!(bb.max, Point::new(5.0, 6.0, 7.0));
    }

    #[test]
    fn test_contains_other() {
        let mut bb1 = BoundingBox::empty();
        bb1.add_point(Point::new(1.0, 2.0, 3.0));
        bb1.add_point(Point::new(4.0, 5.0, 6.0));

        let mut bb2 = BoundingBox::empty();
        bb2.add_point(Point::new(2.0, 3.0, 4.0));
        bb2.add_point(Point::new(3.0, 4.0, 5.0));

        assert!(bb1.contains_other(&bb2));
    }

    #[test]
    fn test_distance() {
        let mut bb1 = BoundingBox::empty();
        bb1.add_point(Point::new(1.0, 2.0, 3.0));

        let mut bb2 = BoundingBox::empty();
        bb2.add_point(Point::new(1.0, 2.0, 2.0));

        assert_eq!(bb1.distance(&bb2), 1.0);
    }
}