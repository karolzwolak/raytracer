use crate::{
    math::{approx_eq::ApproxEq, point::Point, tuple::Tuple, vector::Vector},
    render::ray::{intersection::IntersectionCollector, Ray},
    scene::object::bounding_box::BoundingBox,
};

pub struct PlaneXZ {}

impl PlaneXZ {
    pub fn local_normal_at() -> Vector {
        Vector::new(0., 1., 0.)
    }
    pub fn local_intersect(object_ray: &Ray, collector: &mut IntersectionCollector) {
        let parallel = object_ray.direction().y().approx_eq(&0.);
        if parallel {
            return;
        }
        collector.add(-object_ray.origin().y() / object_ray.direction().y());
    }
    pub fn bounding_box() -> BoundingBox {
        BoundingBox {
            min: Point::new(f64::NEG_INFINITY, 0., f64::NEG_INFINITY),
            max: Point::new(f64::INFINITY, 0., f64::INFINITY),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        assert_approx_eq_low_prec,
        math::{approx_eq::ApproxEq, point::Point, tuple::Tuple, vector::Vector},
        render::ray::Ray,
        scene::object::{primitive::shape::Shape, Object},
    };

    #[test]
    fn ray_intersecting_plane_from_above() {
        let plane = Object::primitive_with_shape(Shape::Plane);
        let ray = Ray::new(Point::new(0., 1., 0.), Vector::new(0., -1., 0.));

        assert_eq!(plane.intersection_times_testing(&ray), vec![1.]);
    }

    #[test]
    fn ray_intersecting_plane_from_below() {
        let plane = Object::primitive_with_shape(Shape::Plane);
        let ray = Ray::new(Point::new(0., -1., 0.), Vector::new(0., 1., 0.));

        assert_eq!(plane.intersection_times_testing(&ray), vec![1.]);
    }

    #[test]
    fn normal_of_plane_is_const_everywhere() {
        let plane = Object::primitive_with_shape(Shape::Plane);

        let expected = Vector::new(0., 1., 0.);

        assert_approx_eq_low_prec!(plane.normal_vector_at(Point::new(0., 0., 0.,)), expected);
        assert_approx_eq_low_prec!(plane.normal_vector_at(Point::new(10., 0., -10.,)), expected);
        assert_approx_eq_low_prec!(plane.normal_vector_at(Point::new(-5., 0., 150.,)), expected);
    }

    #[test]
    fn normal_on_surface_of_cube() {
        let cube = Object::primitive_with_shape(Shape::Cube);
        let examples = vec![
            (Point::new(1., 0.5, -0.8), Vector::new(1., 0., 0.)),
            (Point::new(-1., -0.2, 0.9), Vector::new(-1., 0., 0.)),
            (Point::new(-0.4, 1., -0.1), Vector::new(0., 1., 0.)),
            (Point::new(0.3, -1., -0.7), Vector::new(0., -1., 0.)),
            (Point::new(-0.6, 0.3, 1.), Vector::new(0., 0., 1.)),
            (Point::new(0.4, 0.4, -1.), Vector::new(0., 0., -1.)),
            (Point::new(1., 1., 1.), Vector::new(1., 0., 0.)),
            (Point::new(-1., -1., -1.), Vector::new(-1., 0., 0.)),
        ];

        for (point, expected) in examples {
            assert_approx_eq_low_prec!(cube.normal_vector_at(point), expected);
        }
    }
}
