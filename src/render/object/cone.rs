use crate::{
    approx_eq::{self, ApproxEq},
    primitive::{point::Point, tuple::Tuple, vector::Vector},
    render::{intersection::IntersectionCollector, ray::Ray},
};

use super::bounding_box::BoundingBox;

#[derive(Clone, Debug)]
pub struct Cone {
    pub y_min: f64,
    pub y_max: f64,
    pub closed: bool,
}

impl Cone {
    pub fn new(height: f64, y_offset: f64, closed: bool) -> Self {
        assert!(height >= 0.);

        let (y_min, y_max) = if height.approx_eq(&0.) {
            (f64::NEG_INFINITY, f64::INFINITY)
        } else {
            (-height / 2. + y_offset, height / 2. + y_offset)
        };

        Cone {
            y_min,
            y_max,
            closed,
        }
    }
    pub fn unit_cone() -> Self {
        Self::new(1., -0.5, true)
    }
    pub fn local_normal_at(&self, object_point: Point) -> Vector {
        let dist = object_point.x().powi(2) + object_point.z().powi(2);

        let min_radius = self.y_min.abs();
        let max_radius = self.y_max.abs();

        if dist < max_radius && object_point.y() >= self.y_max - approx_eq::EPSILON {
            Vector::new(0., 1., 0.)
        } else if dist < min_radius && object_point.y() <= self.y_min + approx_eq::EPSILON {
            Vector::new(0., -1., 0.)
        } else {
            let mut y = (object_point.x().powi(2) + object_point.z().powi(2)).sqrt();
            if object_point.y() > 0. {
                y = -y;
            }
            Vector::new(object_point.x(), y, object_point.z())
        }
    }
    pub fn local_intersect(&self, object_ray: &Ray, collector: &mut IntersectionCollector) {
        if self.y_min.approx_eq(&self.y_max) {
            return;
        }
        self.intersect_cone_caps(object_ray, collector);

        let dir = object_ray.direction();
        let origin = object_ray.origin();

        let a = dir.x().powi(2) - dir.y().powi(2) + dir.z().powi(2);
        let b = 2. * (origin.x() * dir.x() - origin.y() * dir.y() + origin.z() * dir.z());
        let c = origin.x().powi(2) - origin.y().powi(2) + origin.z().powi(2);

        let ray_parallel_to_one_half = a.approx_eq(&0.);

        if ray_parallel_to_one_half {
            if b.approx_eq(&0.) {
                return;
            }
            let t = -c / (2. * b);
            collector.add(t);
        } else {
            let discriminant = b * b - 4. * a * c;
            if discriminant < 0. {
                return;
            }
            let delta_sqrt = discriminant.sqrt();
            let t0 = (-b - delta_sqrt) / (2. * a);
            let t1 = (-b + delta_sqrt) / (2. * a);
            let y0 = origin.y() + t0 * dir.y();

            if self.y_min < y0 && y0 < self.y_max {
                collector.add(t0);
            }
            let y1 = origin.y() + t1 * dir.y();
            if self.y_min < y1 && y1 < self.y_max {
                collector.add(t1);
            }
        }
    }
    pub fn bounding_box(&self) -> BoundingBox {
        BoundingBox {
            min: Point::new(self.y_min, self.y_min, self.y_min),
            max: Point::new(self.y_max, self.y_max, self.y_max),
        }
    }

    fn check_cap_within_radius(&self, ray: &Ray, t: f64, radius: f64) -> bool {
        assert!(radius >= 0.);
        let x = ray.origin().x() + t * ray.direction().x();
        let z = ray.origin().z() + t * ray.direction().z();

        x * x + z * z <= radius
    }

    fn intersect_cone_caps(&self, ray: &Ray, collector: &mut IntersectionCollector) {
        if !self.closed || ray.direction().y().approx_eq(&0.) {
            return;
        }
        let tmin = (self.y_min - ray.origin().y()) / ray.direction().y();
        let tmax = (self.y_max - ray.origin().y()) / ray.direction().y();

        if self.check_cap_within_radius(ray, tmin, self.y_min.abs()) {
            collector.add(tmin);
        }
        if self.check_cap_within_radius(ray, tmax, self.y_max.abs()) {
            collector.add(tmax);
        }
    }
}

impl Default for Cone {
    fn default() -> Self {
        Cone {
            y_min: f64::NEG_INFINITY,
            y_max: f64::INFINITY,
            closed: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::SQRT_2;

    use crate::{
        approx_eq::ApproxEq,
        primitive::{point::Point, tuple::Tuple, vector::Vector},
        render::{
            object::{shape::Shape, Object},
            ray::Ray,
        },
    };

    use super::Cone;

    #[test]
    fn intersecting_cone() {
        let cone = Object::with_shape(Shape::default_cone());

        let examples = vec![
            (Point::new(0., 0., -5.), Vector::new(0., 0., 1.), (5., 5.)),
            (
                Point::new(0., 0., -5.),
                Vector::new(1., 1., 1.),
                (8.66025, 8.66025),
            ),
            (
                Point::new(1., 1., -5.),
                Vector::new(-0.5, -1., 1.),
                (4.55006, 49.44994),
            ),
        ];

        for (origin, direction, expected) in examples {
            let ray = Ray::new(origin, direction.normalize());
            let times = cone.intersection_times(&ray);

            assert_eq!(times.len(), 2);
            assert!(times[0].approx_eq(&expected.0));
            assert!(times[1].approx_eq(&expected.1));
        }
    }

    #[test]
    fn intersecting_cone_with_ray_parallel_to_one_half() {
        let cone = Object::with_shape(Shape::default_cone());
        let ray = Ray::new(Point::new(0., 0., -1.), Vector::new(0., 1., 1.).normalize());
        let times = cone.intersection_times(&ray);

        assert_eq!(times.len(), 1);
        assert!(times[0].approx_eq(&0.35355));
    }

    #[test]
    fn intersecting_cone_caps() {
        let cone = Object::with_shape(Shape::Cone(Cone {
            y_min: -0.5,
            y_max: 0.5,
            closed: true,
        }));
        let examples = vec![
            (Point::new(0., 0., -5.), Vector::new(0., 1., 0.), 0),
            (Point::new(0., 0., -0.25), Vector::new(0., 1., 1.), 2),
            (Point::new(0., 0., -0.25), Vector::new(0., 1., 0.), 4),
        ];

        for (origin, direction, expected) in examples {
            let ray = Ray::new(origin, direction.normalize());
            let times = cone.intersection_times(&ray);
            assert_eq!(times.len(), expected);
        }
    }

    #[test]
    fn normal_of_cone_caps() {
        let cone = Object::with_shape(Shape::default_cone());

        let examples = vec![
            (Point::new(0., 0., 0.), Vector::new(0., 0., 0.)),
            (Point::new(1., 1., 1.), Vector::new(1., -SQRT_2, 1.)),
            (Point::new(-1., -1., 0.), Vector::new(-1., 1., 0.)),
        ];

        for (point, expected) in examples {
            assert_eq!(cone.normal_vector_at(point), expected.normalize());
        }
    }
}
