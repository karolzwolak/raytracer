#[cfg(test)]
mod tests {
    use std::{f64::consts::FRAC_1_SQRT_2, f64::consts::PI};

    use crate::{
        primitive::{matrix::Matrix, point::Point, tuple::Tuple, vector::Vector},
        render::{
            object::{shape::Shape, Object},
            ray::Ray,
        },
    };

    #[test]
    fn transformed_sphere() {
        // < -2; 6 >
        let obj = Object::sphere(Point::new(2., 2., 2.), 4.);

        let direction = Vector::new(0., 0., 1.);
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(2., 2., 2.), direction)));
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(2., 2., -2.), direction)));
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(4., 2., -2.), direction)));
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(6., 2., -2.), direction)));
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(-2., 2., -2.), direction)));
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(-1., 2., -2.), direction)));
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(3., -1., -2.), direction)));

        assert!(!obj.is_intersected_by_ray(&Ray::new(Point::new(-1., -2., -2.), direction)));
        assert!(!obj.is_intersected_by_ray(&Ray::new(Point::new(3., -8., -2.), direction)));
        assert!(!obj.is_intersected_by_ray(&Ray::new(Point::new(3., -6., -2.), direction)));
    }

    #[test]
    fn intersect_scaled_sphere() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let obj = Object::with_transformation(Shape::Sphere, Matrix::scaling_uniform(2.));

        assert_eq!(obj.intersection_times(&ray), vec![3., 7.]);
    }
    #[test]
    fn intersect_translated_sphere() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let obj = Object::with_transformation(Shape::Sphere, Matrix::translation(5., 0., 0.));

        assert_eq!(obj.intersection_times(&ray), vec![]);
    }
    #[test]
    fn normal_on_sphere_x_axis() {
        let sphere_obj = Object::with_shape(Shape::Sphere);

        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(1., 0., 0.,)),
            Vector::new(1., 0., 0.)
        );
    }
    #[test]
    fn normal_on_sphere_y_axis() {
        let sphere_obj = Object::with_shape(Shape::Sphere);

        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(0., 1., 0.,)),
            Vector::new(0., 1., 0.)
        );
    }
    #[test]
    fn normal_on_sphere_z_axis() {
        let sphere_obj = Object::with_shape(Shape::Sphere);

        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(0., 0., 1.,)),
            Vector::new(0., 0., 1.)
        );
    }
    #[test]
    fn normal_on_sphere_at_noaxial_point() {
        let sphere_obj = Object::with_shape(Shape::Sphere);

        let frac_sqrt_3_3 = 3_f64.sqrt() / 3.;
        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(frac_sqrt_3_3, frac_sqrt_3_3, frac_sqrt_3_3)),
            Vector::new(frac_sqrt_3_3, frac_sqrt_3_3, frac_sqrt_3_3)
        );
    }
    #[test]
    fn compute_normal_on_translated_sphere() {
        let mut sphere_obj = Object::with_shape(Shape::Sphere);
        sphere_obj.apply_transformation(Matrix::translation(0., 1., 0.));
        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(0., 1. + FRAC_1_SQRT_2, -FRAC_1_SQRT_2)),
            Vector::new(0., FRAC_1_SQRT_2, -FRAC_1_SQRT_2)
        );
    }
    #[test]
    fn compute_normal_on_transformed_sphere() {
        let mut sphere_obj = Object::with_shape(Shape::Sphere);
        sphere_obj.apply_transformation(Matrix::scaling(1., 0.5, 1.) * Matrix::rotation_z(PI / 5.));
        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(0., FRAC_1_SQRT_2, -FRAC_1_SQRT_2)),
            Vector::new(0., 0.97014, -0.24254)
        );
    }
}
