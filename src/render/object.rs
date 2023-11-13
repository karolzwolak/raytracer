use crate::primitive::matrix4::Matrix4;

use super::shape::Shape;

pub struct Object {
    shape: Shape,
    transformation: Matrix4,
}

impl Object {
    pub fn new(shape: Shape) -> Self {
        Self::new_with_transformation(shape, Matrix4::identiy_matrix())
    }
    pub fn new_with_transformation(shape: Shape, matrix: Matrix4) -> Self {
        Self {
            shape,
            transformation: matrix,
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::matrix4::Matrix4;

    #[test]
    fn identiy_matrix_is_obj_default_transformation() {
        assert_eq!(
            Object::new(Shape::Sphere()).transformation,
            Matrix4::identiy_matrix()
        );
    }
}
