use crate::{
    approx_eq,
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

impl BoundingBox {
    pub fn empty() -> Self {
        Self {
            min: Point::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Point::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.min == self.max
            || self.min.x() > self.max.x()
                && self.min.y() > self.max.y()
                && self.min.z() > self.max.z()
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
    pub fn add_bounding_box(&mut self, other: BoundingBox) {
        if other.is_empty() {
            return;
        }
        self.add_point(other.min);
        self.add_point(other.max);
    }
    pub fn transformed(&self, matrix: Matrix) -> Self {
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
    pub fn transform(&mut self, matrix: Matrix) {
        *self = self.transformed(matrix);
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
    pub fn as_object(&self) -> Object {
        // render slightly bigger box to avoid acne effect
        const LEN_FACTOR: f64 = 0.5 * (1. + approx_eq::EPSILON);

        let x_len = self.max.x() - self.min.x();
        let y_len = self.max.y() - self.min.y();
        let z_len = self.max.z() - self.min.z();
        let center = self.center();
        let pattern = Pattern::Const(Color::red());
        Object::new(
            Shape::Cube,
            Material {
                pattern,
                transparency: 0.9,
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
}
