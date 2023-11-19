use crate::{
    primitive::{matrix4::Matrix4, point::Point, tuple::Tuple, vector::Vector},
    transformation::Transform,
};

use super::{intersection::IntersecVec, material::Material, ray::Ray, shape::Shape};

pub struct Object {
    shape: Shape,
    material: Material,
    transformation: Matrix4,
}

impl Object {
    pub fn new(shape: Shape, material: Material, transformation: Matrix4) -> Self {
        Self {
            shape,
            material,
            transformation,
        }
    }

    pub fn with_shape(shape: Shape) -> Self {
        Self::with_transformation(shape, Matrix4::identity_matrix())
    }
    pub fn with_shape_material(shape: Shape, material: Material) -> Self {
        Self::new(shape, material, Matrix4::identity_matrix())
    }
    pub fn with_transformation(shape: Shape, matrix: Matrix4) -> Self {
        Self::new(shape, Material::default(), matrix)
    }
    pub fn sphere(center: Point, radius: f64) -> Self {
        Self::with_transformation(
            Shape::Sphere,
            Matrix4::identity_matrix()
                .scale(radius, radius, radius)
                .translate(center.x(), center.y(), center.z())
                .get_transformed(),
        )
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn transformation(&self) -> &Matrix4 {
        &self.transformation
    }
    pub fn transformation_inverse(&self) -> Option<Matrix4> {
        self.transformation.inverse()
    }
    pub fn apply_transformation(&mut self, matrix: Matrix4) {
        self.transformation = self.transformation * matrix;
    }
    pub fn has_intersection_with_ray(&self, ray: &Ray) -> bool {
        IntersecVec::does_intersect(ray, self)
    }
    pub fn normal_vector_at(&self, world_point: Point) -> Vector {
        let inverse = self.transformation_inverse().unwrap();
        let object_point = inverse * world_point;
        let object_normal = object_point - Point::zero();

        let world_normal = inverse.transpose() * object_normal;
        world_normal.normalize()
    }

    pub fn material(&self) -> &Material {
        &self.material
    }

    pub fn set_material(&mut self, material: Material) {
        self.material = material;
    }

    pub fn material_mut(&mut self) -> &mut Material {
        &mut self.material
    }
}

#[cfg(test)]
mod tests {
    use std::{f64::consts::FRAC_1_SQRT_2, f64::consts::PI};

    use super::*;
    use crate::{
        primitive::{matrix4::Matrix4, vector::Vector},
        transformation::{rotation_z_matrix, scaling_matrix, translation_matrix},
    };

    #[test]
    fn identiy_matrix_is_obj_default_transformation() {
        assert_eq!(
            Object::with_shape(Shape::Sphere).transformation,
            Matrix4::identity_matrix()
        );
    }
    #[test]
    fn transformed_sphere() {
        // < -2; 6 >
        let obj = Object::sphere(Point::new(2., 2., 2.), 4.);

        let direction = Vector::new(0., 0., 1.);
        assert!(obj.has_intersection_with_ray(&Ray::new(Point::new(2., 2., 2.), direction)));
        assert!(obj.has_intersection_with_ray(&Ray::new(Point::new(2., 2., -2.), direction)));
        assert!(obj.has_intersection_with_ray(&Ray::new(Point::new(4., 2., -2.), direction)));
        assert!(obj.has_intersection_with_ray(&Ray::new(Point::new(6., 2., -2.), direction)));
        assert!(obj.has_intersection_with_ray(&Ray::new(Point::new(-2., 2., -2.), direction)));
        assert!(obj.has_intersection_with_ray(&Ray::new(Point::new(-1., 2., -2.), direction)));
        assert!(obj.has_intersection_with_ray(&Ray::new(Point::new(3., -1., -2.), direction)));

        assert!(!obj.has_intersection_with_ray(&Ray::new(Point::new(-1., -2., -2.), direction)));
        assert!(!obj.has_intersection_with_ray(&Ray::new(Point::new(3., -8., -2.), direction)));
        assert!(!obj.has_intersection_with_ray(&Ray::new(Point::new(3., -6., -2.), direction)));
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
    fn normal_is_normalized() {
        let sphere_obj = Object::with_shape(Shape::Sphere);

        let frac_sqrt_3_3 = 3_f64.sqrt() / 3.;
        let normal =
            sphere_obj.normal_vector_at(Point::new(frac_sqrt_3_3, frac_sqrt_3_3, frac_sqrt_3_3));
        assert_eq!(normal, normal.normalize());
    }
    #[test]
    fn compute_normal_on_translated_sphere() {
        let mut sphere_obj = Object::with_shape(Shape::Sphere);
        sphere_obj.apply_transformation(translation_matrix(0., 1., 0.));
        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(0., 1. + FRAC_1_SQRT_2, -FRAC_1_SQRT_2)),
            Vector::new(0., FRAC_1_SQRT_2, -FRAC_1_SQRT_2)
        );
    }
    #[test]
    fn compute_normal_on_transformed_sphere() {
        let mut sphere_obj = Object::with_shape(Shape::Sphere);
        sphere_obj.apply_transformation(scaling_matrix(1., 0.5, 1.) * rotation_z_matrix(PI / 5.));
        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(0., FRAC_1_SQRT_2, -FRAC_1_SQRT_2)),
            Vector::new(0., 0.97014, -0.24254)
        );
    }
}
