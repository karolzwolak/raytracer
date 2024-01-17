use crate::primitive::{
    matrix::{Matrix, Transform},
    point::Point,
    tuple::Tuple,
    vector::Vector,
};

use super::{intersection::IntersecVec, material::Material, ray::Ray};

#[derive(Copy, Clone)]
pub enum Shape {
    /// Unit sphere at point zero
    Sphere,
    /// Plane extending in x and z directions, at y = 0
    Plane,
}

impl Shape {
    pub fn object_normal_at(&self, object_point: Point) -> Vector {
        match self {
            Shape::Sphere => object_point - Point::zero(),
            Shape::Plane => Vector::new(0., 1., 0.),
        }
    }
}

#[derive(Clone)]
pub struct Object {
    shape: Shape,
    material: Material,
    transformation: Matrix,
}

impl Object {
    pub fn new(shape: Shape, material: Material, transformation: Matrix) -> Self {
        Self {
            shape,
            material,
            transformation,
        }
    }

    pub fn with_shape(shape: Shape) -> Self {
        Self::with_transformation(shape, Matrix::identity())
    }
    pub fn with_shape_material(shape: Shape, material: Material) -> Self {
        Self::new(shape, material, Matrix::identity())
    }
    pub fn with_transformation(shape: Shape, matrix: Matrix) -> Self {
        Self::new(shape, Material::default(), matrix)
    }
    pub fn sphere(center: Point, radius: f64) -> Self {
        Self::with_transformation(
            Shape::Sphere,
            Matrix::identity()
                .scale(radius, radius, radius)
                .translate(center.x(), center.y(), center.z())
                .transformed(),
        )
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn transformation(&self) -> &Matrix {
        &self.transformation
    }
    pub fn transformation_inverse(&self) -> Option<Matrix> {
        self.transformation.inverse()
    }
    pub fn apply_transformation(&mut self, matrix: Matrix) {
        self.transformation = self.transformation * matrix;
    }
    pub fn has_intersection_with_ray(&self, ray: &Ray) -> bool {
        IntersecVec::does_intersect(ray, self)
    }
    pub fn normal_vector_at(&self, world_point: Point) -> Vector {
        let inverse = self.transformation_inverse().unwrap();
        let object_point = inverse * world_point;

        let object_normal = self.shape.object_normal_at(object_point);
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
    use crate::primitive::{matrix::Matrix, vector::Vector};

    #[test]
    fn identiy_matrix_is_obj_default_transformation() {
        assert_eq!(
            Object::with_shape(Shape::Sphere).transformation,
            Matrix::identity()
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

    #[test]
    fn normal_of_plane_is_const_everywhere() {
        let plane = Object::with_shape(Shape::Plane);

        let expected = Vector::new(0., 1., 0.);

        assert_eq!(plane.normal_vector_at(Point::new(0., 0., 0.,)), expected);
        assert_eq!(plane.normal_vector_at(Point::new(10., 0., -10.,)), expected);
        assert_eq!(plane.normal_vector_at(Point::new(-5., 0., 150.,)), expected);
    }
}
