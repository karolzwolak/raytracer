use crate::{
    math::{
        approx_eq::{self, ApproxEq},
        point::Point,
        tuple::Tuple,
        vector::Vector,
    },
    render::ray::{intersection::IntersectionCollector, Ray},
    scene::object::bounding_box::BoundingBox,
};

/// Cylinder with radius 1
#[derive(Clone, Debug, PartialEq)]
pub struct Cylinder {
    pub y_min: f64,
    pub y_max: f64,
    pub closed: bool,
}

impl Cylinder {
    pub const CYLINDER_RADIUS: f64 = 1.;
    pub fn new(y_min: f64, y_max: f64, closed: bool) -> Self {
        Self {
            y_min,
            y_max,
            closed,
        }
    }
    pub fn with_height(height: f64, closed: bool) -> Self {
        assert!(height >= 0.);
        let (y_min, y_max) = if height.approx_eq(&0.) {
            (f64::NEG_INFINITY, f64::INFINITY)
        } else {
            (-height / 2., height / 2.)
        };
        Self {
            y_min,
            y_max,
            closed,
        }
    }
    pub fn local_normal_at(&self, object_point: Point) -> Vector {
        let dist = object_point.x().powi(2) + object_point.z().powi(2);
        let point_within_radius = dist < Self::CYLINDER_RADIUS;

        if point_within_radius && object_point.y() >= self.y_max - approx_eq::EPSILON {
            Vector::new(0., 1., 0.)
        } else if point_within_radius && object_point.y() <= self.y_min + approx_eq::EPSILON {
            Vector::new(0., -1., 0.)
        } else {
            Vector::new(object_point.x(), 0., object_point.z())
        }
    }
    pub fn local_intersect(&self, object_ray: &Ray, collector: &mut IntersectionCollector) {
        if self.y_min.approx_eq(&self.y_max) {
            return;
        }

        self.intersect_cyl_caps(object_ray, collector);

        let a = object_ray.direction().x().powi(2) + object_ray.direction().z().powi(2);

        // ray is parallel to the y axis
        if a.approx_eq(&0.) {
            return;
        }

        let b = 2. * object_ray.origin().x() * object_ray.direction().x()
            + 2. * object_ray.origin().z() * object_ray.direction().z();
        let c = object_ray.origin().x().powi(2) + object_ray.origin().z().powi(2) - 1.;

        let discriminant = b * b - 4. * a * c;

        if discriminant < 0. {
            return;
        }

        let delta_sqrt = discriminant.sqrt();

        let t0 = (-b - delta_sqrt) / (2. * a);
        let t1 = (-b + delta_sqrt) / (2. * a);

        let y0 = object_ray.origin().y() + t0 * object_ray.direction().y();

        if self.y_min < y0 && y0 < self.y_max {
            collector.add(t0);
        }

        let y1 = object_ray.origin().y() + t1 * object_ray.direction().y();

        if self.y_min < y1 && y1 < self.y_max {
            collector.add(t1);
        }
    }
    pub fn bounding_box(&self) -> BoundingBox {
        BoundingBox {
            min: Point::new(-1., self.y_min, -1.),
            max: Point::new(1., self.y_max, 1.),
        }
    }
    fn check_cap_within_radius(&self, ray: &Ray, t: f64) -> bool {
        let x = ray.origin().x() + t * ray.direction().x();
        let z = ray.origin().z() + t * ray.direction().z();

        x * x + z * z <= 1.
    }
    fn intersect_cyl_caps(&self, ray: &Ray, collector: &mut IntersectionCollector) {
        if !self.closed || ray.direction().y().approx_eq(&0.) {
            return;
        }
        let tmin = (self.y_min - ray.origin().y()) / ray.direction().y();
        let tmax = (self.y_max - ray.origin().y()) / ray.direction().y();

        if self.check_cap_within_radius(ray, tmin) {
            collector.add(tmin);
        }
        if self.check_cap_within_radius(ray, tmax) {
            collector.add(tmax);
        }
    }
}

impl Default for Cylinder {
    fn default() -> Self {
        Self {
            y_min: f64::NEG_INFINITY,
            y_max: f64::INFINITY,
            closed: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Cylinder;
    use crate::{
        assert_approx_eq_low_prec,
        math::{approx_eq::ApproxEq, point::Point, tuple::Tuple, vector::Vector},
        render::ray::Ray,
        scene::object::{primitive::shape::Shape, Object},
    };

    #[test]
    fn ray_misses_cylinder() {
        let cyl = Object::primitive_with_shape(Shape::default_cylinder());
        let examples = vec![
            Ray::new(Point::new(1., 0., 0.), Vector::new(0., 1., 0.)),
            Ray::new(Point::new(0., 0., 0.), Vector::new(0., 1., 0.)),
            Ray::new(Point::new(0., 0., -5.), Vector::new(1., 1., 1.)),
        ];

        for ray in examples {
            assert!(!cyl.is_intersected_by_ray(&ray));
        }
    }

    #[test]
    fn ray_intersects_cylinder() {
        let cyl = Object::primitive_with_shape(Shape::default_cylinder());

        let examples = vec![
            (
                Point::new(1., 0., -5.),
                Vector::new(0., 0., 1.),
                vec![5., 5.],
            ),
            (
                Point::new(0., 0., -5.),
                Vector::new(0., 0., 1.),
                vec![4., 6.],
            ),
            (
                Point::new(0.5, 0., -5.),
                Vector::new(0.1, 1., 1.),
                vec![6.80798, 7.08872],
            ),
        ];

        for (origin, direction, expected) in examples {
            let ray = Ray::new(origin, direction.normalize());
            let times = cyl.intersection_times_testing(&ray);

            assert_eq!(times.len(), expected.len());
            for t in times.iter().zip(expected.iter()) {
                assert_approx_eq_low_prec!(t.0, t.1);
            }
        }
    }

    #[test]
    fn normal_of_cylinder() {
        let cyl = Object::primitive_with_shape(Shape::default_cylinder());

        let examples = vec![
            (Point::new(1., 0., 0.), Vector::new(1., 0., 0.)),
            (Point::new(0., 5., -1.), Vector::new(0., 0., -1.)),
            (Point::new(0., -2., 1.), Vector::new(0., 0., 1.)),
            (Point::new(-1., 1., 0.), Vector::new(-1., 0., 0.)),
        ];

        for (point, expected) in examples {
            assert_approx_eq_low_prec!(cyl.normal_vector_at(point), expected);
        }
    }

    fn get_cylinder() -> Object {
        Object::primitive_with_shape(Shape::Cylinder(Cylinder {
            y_min: 1.,
            y_max: 2.,
            closed: false,
        }))
    }

    #[test]
    fn default_min_max_for_cylinder() {
        let cyl = Cylinder::default();

        assert_approx_eq_low_prec!(cyl.y_min, f64::NEG_INFINITY);
        assert_approx_eq_low_prec!(cyl.y_max, f64::INFINITY);
        assert!(!cyl.closed);
    }

    #[test]
    fn intersecting_constrained_cylinder() {
        let cyl = get_cylinder();
        let examples = vec![
            (Point::new(0., 1.5, 0.), Vector::new(0.1, 1., 0.), 0),
            (Point::new(0., 3., -5.), Vector::new(0., 0., 1.), 0),
            (Point::new(0., 0., -5.), Vector::new(0., 0., 1.), 0),
            (Point::new(0., 2., -5.), Vector::new(0., 0., 1.), 0),
            (Point::new(0., 1., -5.), Vector::new(0., 0., 1.), 0),
            (Point::new(0., 1.5, -2.), Vector::new(0., 0., 1.), 2),
        ];

        for (origin, direction, expected) in examples {
            let ray = Ray::new(origin, direction.normalize());
            let times = cyl.intersection_times_testing(&ray);
            assert_eq!(times.len(), expected);
        }
    }

    #[test]
    fn intersecting_cylinder_end_caps() {
        let cyl = Object::primitive_with_shape(Shape::Cylinder(Cylinder {
            y_min: 1.,
            y_max: 2.,
            closed: true,
        }));

        let examples = vec![
            (Point::new(0., 3., 0.), Vector::new(0., -1., 0.), 2),
            (Point::new(0., 3., -2.), Vector::new(0., -1., 2.), 2),
            (Point::new(0., 4., -2.), Vector::new(0., -1., 1.), 2), // corner case
            (Point::new(0., 0., -2.), Vector::new(0., 1., 2.), 2),
            (Point::new(0., -1., -2.), Vector::new(0., 1., 1.), 2), // corner case
        ];

        for (origin, direction, expected) in examples {
            let ray = Ray::new(origin, direction.normalize());
            let times = cyl.intersection_times_testing(&ray);
            assert_eq!(times.len(), expected);
        }
    }

    #[test]
    fn normal_of_cylinder_end_caps() {
        let cyl = get_cylinder();

        let examples = vec![
            (Point::new(0., 1., 0.), Vector::new(0., -1., 0.)),
            (Point::new(0.5, 1., 0.), Vector::new(0., -1., 0.)),
            (Point::new(0., 1., 0.5), Vector::new(0., -1., 0.)),
            (Point::new(0., 2., 0.), Vector::new(0., 1., 0.)),
            (Point::new(0.5, 2., 0.), Vector::new(0., 1., 0.)),
            (Point::new(0., 2., 0.5), Vector::new(0., 1., 0.)),
        ];

        for (point, expected) in examples {
            assert_approx_eq_low_prec!(cyl.normal_vector_at(point), expected);
        }
    }
}
