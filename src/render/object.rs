use crate::{
    primitive::{matrix4::Matrix4, point::Point, tuple::Tuple, vector::Vector},
    transformation::Transform,
};

use super::{intersection::IntersecVec, ray::Ray, shape::Shape};

pub struct Object {
    shape: Shape,
    transformation: Matrix4,
}

impl Object {
    pub fn new(shape: Shape) -> Self {
        Self::new_with_transformation(shape, Matrix4::identity_matrix())
    }
    pub fn new_with_transformation(shape: Shape, matrix: Matrix4) -> Self {
        Self {
            shape,
            transformation: matrix,
        }
    }
    pub fn new_sphere(center: Point, radius: f64) -> Self {
        Self::new_with_transformation(
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{matrix4::Matrix4, vector::Vector};

    #[test]
    fn identiy_matrix_is_obj_default_transformation() {
        assert_eq!(
            Object::new(Shape::Sphere).transformation,
            Matrix4::identity_matrix()
        );
    }
    #[test]
    fn transformed_sphere() {
        // < -2; 6 >
        let obj = Object::new_sphere(Point::new(2., 2., 2.), 4.);

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
}
