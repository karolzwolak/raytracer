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
        let cone = Object::with_shape(Shape::Cone {
            y_min: -0.5,
            y_max: 0.5,
            closed: true,
        });
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
