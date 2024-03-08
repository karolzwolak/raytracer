#[cfg(test)]
mod tests {
    use crate::{
        approx_eq::ApproxEq,
        primitive::{point::Point, tuple::Tuple, vector::Vector},
        render::{
            object::{shape::Shape, Object},
            ray::Ray,
        },
    };

    #[test]
    fn ray_misses_cylinder() {
        let cyl = Object::with_shape(Shape::default_cylinder());
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
        let cyl = Object::with_shape(Shape::default_cylinder());

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
            let times = cyl.intersection_times(&ray);

            assert_eq!(times.len(), expected.len());
            for t in times.iter().zip(expected.iter()) {
                assert!(t.0.approx_eq(t.1));
            }
        }
    }

    #[test]
    fn normal_of_cylinder() {
        let cyl = Object::with_shape(Shape::default_cylinder());

        let examples = vec![
            (Point::new(1., 0., 0.), Vector::new(1., 0., 0.)),
            (Point::new(0., 5., -1.), Vector::new(0., 0., -1.)),
            (Point::new(0., -2., 1.), Vector::new(0., 0., 1.)),
            (Point::new(-1., 1., 0.), Vector::new(-1., 0., 0.)),
        ];

        for (point, expected) in examples {
            assert_eq!(cyl.normal_vector_at(point), expected);
        }
    }

    #[test]
    fn default_min_max_for_cylinder() {
        let cyl = Shape::default_cylinder();

        if let Shape::Cylinder {
            y_min,
            y_max,
            closed,
        } = cyl
        {
            assert_eq!(y_min, f64::NEG_INFINITY);
            assert_eq!(y_max, f64::INFINITY);
            assert!(!closed);
        } else {
            panic!("Expected cylinder");
        }
    }

    #[test]
    fn intersecting_constrained_cylinder() {
        let cyl = Object::with_shape(Shape::Cylinder {
            y_min: 1.,
            y_max: 2.,
            closed: false,
        });

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
            let times = cyl.intersection_times(&ray);
            assert_eq!(times.len(), expected);
        }
    }

    #[test]
    fn intersecting_cylinder_end_caps() {
        let cyl = Object::with_shape(Shape::Cylinder {
            y_min: 1.,
            y_max: 2.,
            closed: true,
        });

        let examples = vec![
            (Point::new(0., 3., 0.), Vector::new(0., -1., 0.), 2),
            (Point::new(0., 3., -2.), Vector::new(0., -1., 2.), 2),
            (Point::new(0., 4., -2.), Vector::new(0., -1., 1.), 2), // corner case
            (Point::new(0., 0., -2.), Vector::new(0., 1., 2.), 2),
            (Point::new(0., -1., -2.), Vector::new(0., 1., 1.), 2), // corner case
        ];

        for (origin, direction, expected) in examples {
            let ray = Ray::new(origin, direction.normalize());
            let times = cyl.intersection_times(&ray);
            assert_eq!(times.len(), expected);
        }
    }

    #[test]
    fn normal_of_cylinder_end_caps() {
        let cyl = Object::with_shape(Shape::Cylinder {
            y_min: 1.,
            y_max: 2.,
            closed: true,
        });

        let examples = vec![
            (Point::new(0., 1., 0.), Vector::new(0., -1., 0.)),
            (Point::new(0.5, 1., 0.), Vector::new(0., -1., 0.)),
            (Point::new(0., 1., 0.5), Vector::new(0., -1., 0.)),
            (Point::new(0., 2., 0.), Vector::new(0., 1., 0.)),
            (Point::new(0.5, 2., 0.), Vector::new(0., 1., 0.)),
            (Point::new(0., 2., 0.5), Vector::new(0., 1., 0.)),
        ];

        for (point, expected) in examples {
            assert_eq!(cyl.normal_vector_at(point), expected);
        }
    }
}
