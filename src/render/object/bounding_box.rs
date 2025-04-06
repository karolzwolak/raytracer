use crate::{
    approx_eq::{self, ApproxEq},
    primitive::{
        matrix::{Matrix, Transform},
        point::Point,
        tuple::{Axis, Tuple},
        vector::Vector,
    },
    render::{
        color::Color,
        material::Material,
        object::{shape::Shape, Object},
        pattern::Pattern,
        ray::Ray,
        world::World,
    },
};

pub trait Bounded {
    fn bounding_box(&self) -> &BoundingBox;
}

#[derive(Clone, Debug, PartialEq)]
/// Axis-aligned bounding box
pub struct BoundingBox {
    pub min: Point,
    pub max: Point,
}

impl Bounded for BoundingBox {
    fn bounding_box(&self) -> &BoundingBox {
        self
    }
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

#[cfg(test)]
impl BoundingBox {
    pub fn unit() -> Self {
        Self {
            min: Point::new(-0.5, -0.5, -0.5),
            max: Point::new(0.5, 0.5, 0.5),
        }
    }
}

impl BoundingBox {
    const MAX_DIM: f64 = World::MAX_DIM;
    pub fn empty() -> Self {
        Self {
            min: Point::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Point::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.min.x() == f64::INFINITY
            && self.min.y() == f64::INFINITY
            && self.min.z() == f64::INFINITY
            && self.max.x() == f64::NEG_INFINITY
            && self.max.y() == f64::NEG_INFINITY
            && self.max.z() == f64::NEG_INFINITY
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
    pub fn limit_dimensions(&mut self) {
        self.max.limit_upper(Self::MAX_DIM);
        self.min.limit_lower(-Self::MAX_DIM);
        let inf = self.is_infinitely_large();
        if inf {
            println!("{:?}", self);
        }
        assert!(!self.is_infinitely_large());
    }
    pub fn is_infinitely_large(&self) -> bool {
        !self.is_empty() && (self.max - self.min).magnitude() == f64::INFINITY
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
    pub fn intersection_time(&self, ray: &Ray) -> Option<f64> {
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
            return None;
        }
        let (ztmin, ztmax) = self.axis_intersection_times(
            ray.origin().z(),
            ray.dir_inv().z(),
            self.min.z(),
            self.max.z(),
        );

        let tmin = tmin.max(ztmin);
        let tmax = tmax.min(ztmax);

        if tmin > tmax {
            return None;
        }
        Some(tmin)
    }
    pub fn intersection_time_from_point(&self, point: Point) -> Option<f64> {
        let ray = Ray::new(point, (self.center() - point).normalize());
        self.intersection_time(&ray)
    }

    // Returns Vector for easy and explicit access to each axis
    pub fn size(&self) -> Vector {
        self.max - self.min
    }

    pub fn longest_axis(&self) -> (Axis, f64) {
        let size = self.size();
        let longest_len = size.x().max(size.y()).max(size.z());

        let axis = match longest_len {
            len if len == size.x() => Axis::X,
            len if len == size.y() => Axis::Y,
            len if len == size.z() => Axis::Z,
            _ => unreachable!(),
        };

        (axis, longest_len)
    }

    pub fn split_along_longest_axis(&self) -> (BoundingBox, BoundingBox) {
        let (mut x0, mut y0, mut z0) = (self.min.x(), self.min.y(), self.min.z());
        let (mut x1, mut y1, mut z1) = (self.max.x(), self.max.y(), self.max.z());

        let (axis, len) = self.longest_axis();
        match axis {
            Axis::X => {
                x0 += len / 2.;
                x1 = x0;
            }
            Axis::Y => {
                y0 += len / 2.;
                y1 = y0;
            }
            Axis::Z => {
                z0 += len / 2.;
                z1 = z0;
            }
        };
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
    pub fn half_area(&self) -> f64 {
        if self.is_empty() {
            return 0.;
        }
        let x_len = self.max.x() - self.min.x();
        let y_len = self.max.y() - self.min.y();
        let z_len = self.max.z() - self.min.z();
        x_len * y_len + y_len * z_len + z_len * x_len
    }
    pub fn length_vec(&self) -> Vector {
        self.max - self.min
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_approx_eq_low_prec, primitive::matrix::Matrix};

    #[test]
    fn test_empty() {
        let bb = BoundingBox::empty();
        assert_approx_eq_low_prec!(
            bb.min,
            Point::new(f64::INFINITY, f64::INFINITY, f64::INFINITY)
        );
        assert_approx_eq_low_prec!(
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
        assert_approx_eq_low_prec!(bb.min, Point::new(1.0, 2.0, 3.0));
        assert_approx_eq_low_prec!(bb.max, Point::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_add_bounding_box() {
        let mut bb1 = BoundingBox::empty();
        bb1.add_point(Point::new(1.0, 2.0, 3.0));
        let mut bb2 = BoundingBox::empty();
        bb2.add_point(Point::new(4.0, 5.0, 6.0));
        bb1.add_bounding_box(&bb2);
        assert_approx_eq_low_prec!(bb1.min, Point::new(1.0, 2.0, 3.0));
        assert_approx_eq_low_prec!(bb1.max, Point::new(4.0, 5.0, 6.0));
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

        assert_approx_eq_low_prec!(bb.min, Point::new(2.0, 3.0, 4.0));
        assert_approx_eq_low_prec!(bb.max, Point::new(5.0, 6.0, 7.0));
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

        assert_approx_eq_low_prec!(bb1.distance(&bb2), 1.0);
    }
}
