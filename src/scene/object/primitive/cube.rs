use crate::{
    core::{point::Point, tuple::Tuple, vector::Vector},
    render::{intersection::IntersectionCollector, ray::Ray},
};

use crate::BoundingBox;

pub struct UnitCube {}

impl UnitCube {
    pub fn local_normal_at(object_point: Point) -> Vector {
        let abs_x = object_point.x().abs();
        let abs_y = object_point.y().abs();
        let abs_z = object_point.z().abs();
        let max = abs_x.max(abs_y).max(abs_z);

        if max == abs_x {
            Vector::new(object_point.x(), 0., 0.)
        } else if max == abs_y {
            Vector::new(0., object_point.y(), 0.)
        } else {
            Vector::new(0., 0., object_point.z())
        }
    }

    pub fn local_intersect(object_ray: &Ray, collector: &mut IntersectionCollector) {
        let (xtmin, xtmax) =
            Self::cube_axis_intersec_times(object_ray.origin().x(), object_ray.dir_inv().x());
        let (ytmin, ytmax) =
            Self::cube_axis_intersec_times(object_ray.origin().y(), object_ray.dir_inv().y());
        let (ztmin, ztmax) =
            Self::cube_axis_intersec_times(object_ray.origin().z(), object_ray.dir_inv().z());

        let tmin = xtmin.max(ytmin).max(ztmin);
        let tmax = xtmax.min(ytmax).min(ztmax);

        if tmin > tmax {
            return;
        }

        collector.add(tmin);
        collector.add(tmax);
    }

    pub fn bounding_box() -> BoundingBox {
        BoundingBox {
            min: Point::new(-1., -1., -1.),
            max: Point::new(1., 1., 1.),
        }
    }

    fn cube_axis_intersec_times(origin: f64, dir_inv: f64) -> (f64, f64) {
        let tmin_numerator = -1. - origin;
        let tmax_numerator = 1. - origin;

        let tmin = tmin_numerator * dir_inv;
        let tmax = tmax_numerator * dir_inv;

        if tmin < tmax {
            (tmin, tmax)
        } else {
            (tmax, tmin)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        core::{point::Point, tuple::Tuple, vector::Vector},
        render::ray::Ray,
        Object, Shape,
    };

    #[test]
    fn ray_intersects_cube() {
        let cube = Object::core_with_shape(Shape::Cube);
        let examples = vec![
            Ray::new(Point::new(5., 0.5, 0.), Vector::new(-1., 0., 0.)),
            Ray::new(Point::new(-5., 0.5, 0.), Vector::new(1., 0., 0.)),
            Ray::new(Point::new(0.5, 5., 0.), Vector::new(0., -1., 0.)),
            Ray::new(Point::new(0.5, -5., 0.), Vector::new(0., 1., 0.)),
            Ray::new(Point::new(0.5, 0., 5.), Vector::new(0., 0., -1.)),
            Ray::new(Point::new(0.5, 0., -5.), Vector::new(0., 0., 1.)),
            Ray::new(Point::new(0., 0.5, 0.), Vector::new(0., 0., 1.)),
        ];
        let expected_times = [
            vec![4., 6.],
            vec![4., 6.],
            vec![4., 6.],
            vec![4., 6.],
            vec![4., 6.],
            vec![4., 6.],
            vec![-1., 1.],
        ];
        assert_eq!(examples.len(), expected_times.len());

        for (ray, expected) in examples.iter().zip(expected_times.iter()) {
            assert_eq!(cube.intersection_times_testing(ray), *expected);
        }
    }

    #[test]
    fn ray_misses_cube() {
        let cube = Object::core_with_shape(Shape::Cube);

        let rays = vec![
            Ray::new(Point::new(-2., 0., 0.), Vector::new(0.2673, 0.5345, 0.8018)),
            Ray::new(Point::new(0., -2., 0.), Vector::new(0.8018, 0.2673, 0.5345)),
            Ray::new(Point::new(0., 0., -2.), Vector::new(0.5345, 0.8018, 0.2673)),
            Ray::new(Point::new(2., 0., 2.), Vector::new(0., 0., -1.)),
            Ray::new(Point::new(0., 2., 2.), Vector::new(0., -1., 0.)),
            Ray::new(Point::new(2., 2., 0.), Vector::new(-1., 0., 0.)),
        ];

        for ray in rays {
            assert!(!cube.is_intersected_by_ray(&ray));
        }
    }
}
