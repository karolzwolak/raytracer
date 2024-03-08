#[cfg(test)]
mod tests {
    use crate::{
        primitive::{point::Point, tuple::Tuple, vector::Vector},
        render::{
            object::{shape::Shape, Object},
            ray::Ray,
        },
    };

    #[test]
    fn ray_intersects_cube() {
        let cube = Object::with_shape(Shape::Cube);
        let examples = vec![
            Ray::new(Point::new(5., 0.5, 0.), Vector::new(-1., 0., 0.)),
            Ray::new(Point::new(-5., 0.5, 0.), Vector::new(1., 0., 0.)),
            Ray::new(Point::new(0.5, 5., 0.), Vector::new(0., -1., 0.)),
            Ray::new(Point::new(0.5, -5., 0.), Vector::new(0., 1., 0.)),
            Ray::new(Point::new(0.5, 0., 5.), Vector::new(0., 0., -1.)),
            Ray::new(Point::new(0.5, 0., -5.), Vector::new(0., 0., 1.)),
            Ray::new(Point::new(0., 0.5, 0.), Vector::new(0., 0., 1.)),
        ];
        let expected_times = vec![
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
            assert_eq!(cube.intersection_times(ray), *expected);
        }
    }

    #[test]
    fn ray_misses_cube() {
        let cube = Object::with_shape(Shape::Cube);

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
