pub mod bounding_box;
pub mod cone;
pub mod cube;
pub mod cylinder;
pub mod group;
pub mod plane;
pub mod shape;
pub mod smooth_triangle;
pub mod sphere;
pub mod triangle;

use crate::primitive::{
    matrix::{Matrix, Transform},
    point::Point,
    tuple::Tuple,
    vector::Vector,
};

use self::{bounding_box::BoundingBox, group::ObjectGroup, shape::Shape};

use super::{
    intersection::{Intersection, IntersectionCollector},
    material::Material,
    ray::Ray,
};

#[derive(Clone, Debug)]
pub struct Object {
    shape: Shape,
    material: Material,
    transformation: Matrix,
    transformation_inverse: Matrix,
}

impl Object {
    pub fn new(shape: Shape, material: Material, transformation: Matrix) -> Self {
        Self {
            shape,
            material,
            transformation,
            transformation_inverse: transformation
                .inverse()
                .expect("Object with singular tranfromation matrix cannot be rendered"),
        }
    }

    pub fn group(children: Vec<Object>, transformation: Matrix) -> Self {
        Self::with_shape(Shape::Group(ObjectGroup::with_transformations(
            children,
            transformation,
        )))
    }

    pub fn bounding_box(&self) -> BoundingBox {
        self.shape.bounding_box().transformed(self.transformation)
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
    pub fn transformation_inverse(&self) -> Matrix {
        self.transformation_inverse
    }
    pub fn apply_transformation(&mut self, matrix: Matrix) {
        match &mut self.shape {
            Shape::Group(group) => group.apply_transformation(matrix),
            _ => {
                self.transformation = matrix * self.transformation;
                self.transformation_inverse = self.transformation.inverse().unwrap();
            }
        }
    }
    pub fn normal_vector_at(&self, world_point: Point) -> Vector {
        self.normal_vector_at_with_intersection(world_point, None)
    }
    pub fn normal_vector_at_with_intersection<'a>(
        &self,
        world_point: Point,
        i: Option<&'a Intersection<'a>>,
    ) -> Vector {
        let inverse = self.transformation_inverse();
        let object_point = inverse * world_point;

        let object_normal = self.shape.local_normal_at(object_point, i);
        let world_normal = inverse.transpose() * object_normal;
        world_normal.normalize()
    }

    pub fn intersect_with_collector<'a>(
        &'a self,
        world_ray: &Ray,
        collector: &mut IntersectionCollector<'a>,
    ) {
        match &self.shape {
            Shape::Group(ref group) => group.intersect(world_ray, collector),
            _ => {
                collector.set_next_object(self);
                self.shape
                    .local_intersect(&world_ray.transform(self.transformation_inverse), collector);
            }
        }
    }

    pub fn intersect_to_vec<'a>(&'a self, world_ray: &Ray) -> Vec<Intersection<'a>> {
        let mut collector = IntersectionCollector::with_next_object(self);
        self.intersect_with_collector(world_ray, &mut collector);
        collector.collect_sorted()
    }

    pub fn intersection_times(&self, world_ray: &Ray) -> Vec<f64> {
        self.intersect_to_vec(world_ray)
            .iter_mut()
            .map(|i| i.time())
            .collect()
    }

    pub fn is_intersected_by_ray(&self, ray: &Ray) -> bool {
        !self.intersect_to_vec(ray).is_empty()
    }
    pub fn material(&self) -> &Material {
        &self.material
    }

    pub fn set_material(&mut self, material: Material) {
        if let Shape::Group(group) = &mut self.shape {
            group.set_material(material.clone())
        }
        self.material = material;
    }

    pub fn material_mut(&mut self) -> &mut Material {
        &mut self.material
    }

    pub fn get_group(&self) -> Option<&ObjectGroup> {
        self.shape.as_group()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn identiy_matrix_is_obj_default_transformation() {
        assert_eq!(
            Object::with_shape(Shape::Sphere).transformation_inverse(),
            Matrix::identity()
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
}
