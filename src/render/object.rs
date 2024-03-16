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
pub enum Object {
    Primitive(Box<PrimitiveObject>),
    Group(ObjectGroup),
}

impl From<PrimitiveObject> for Object {
    fn from(obj: PrimitiveObject) -> Self {
        Self::from_primitive(obj)
    }
}

impl From<ObjectGroup> for Object {
    fn from(group: ObjectGroup) -> Self {
        Self::from_group(group)
    }
}

impl Transform for Object {
    fn transform(&mut self, matrix: &Matrix) {
        match self {
            Self::Primitive(obj) => obj.transform(matrix),
            Self::Group(group) => group.transform(matrix),
        }
    }

    fn transform_new(&self, matrix: &Matrix) -> Self {
        let mut new = self.clone();
        new.transform(matrix);
        new
    }
}

impl Object {
    pub fn group_with_children(children: Vec<Object>) -> Self {
        Self::from_group(ObjectGroup::new(children))
    }

    pub fn from_group(group: ObjectGroup) -> Self {
        Self::Group(group)
    }

    pub fn from_primitive(obj: PrimitiveObject) -> Self {
        Self::Primitive(Box::new(obj))
    }

    pub fn primitive(shape: Shape, material: Material, transformation: Matrix) -> Self {
        Self::from_primitive(PrimitiveObject::new(shape, material, transformation))
    }

    pub fn primitive_with_shape(shape: Shape) -> Self {
        Self::from_primitive(PrimitiveObject::with_shape(shape))
    }

    pub fn primitive_with_transformation(shape: Shape, transformation: Matrix) -> Self {
        Self::from_primitive(PrimitiveObject::with_transformation(shape, transformation))
    }

    pub fn normal_vector_at(&self, world_point: Point) -> Vector {
        self.normal_vector_at_with_intersection(world_point, None)
    }

    pub fn normal_vector_at_with_intersection<'a>(
        &self,
        world_point: Point,
        i: Option<&'a Intersection<'a>>,
    ) -> Vector {
        match self {
            Self::Primitive(obj) => obj.normal_vector_at_with_intersection(world_point, i),
            Self::Group(_) => todo!(),
        }
    }

    pub fn intersect<'a>(&'a self, world_ray: &Ray, collector: &mut IntersectionCollector<'a>) {
        collector.set_next_object(self);
        match self {
            Self::Primitive(obj) => obj.intersect_with_collector(world_ray, collector),
            Self::Group(group) => group.intersect(world_ray, collector),
        }
    }

    pub fn intersect_to_vec<'a>(&'a self, world_ray: &Ray) -> Vec<Intersection<'a>> {
        let mut collector = IntersectionCollector::with_next_object(self);
        self.intersect(world_ray, &mut collector);
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

    pub fn material(&self) -> Option<&Material> {
        self.as_primitive().map(|p| p.material())
    }

    pub fn material_mut(&mut self) -> Option<&mut Material> {
        self.as_primitive_mut().map(|p| p.material_mut())
    }

    pub fn set_material(&mut self, material: Material) {
        match self {
            Self::Primitive(obj) => obj.set_material(material),
            Self::Group(group) => group.set_material(material),
        }
    }

    pub fn material_unwrapped(&self) -> &Material {
        self.material().expect("Object has no material")
    }

    pub fn normalize_and_center(&mut self) {
        let bb = self.bounding_box();
        let center = bb.center();
        let diff = bb.max - bb.min;
        let length = diff.x().max(diff.y()).max(diff.z());

        self.transform(
            Matrix::identity()
                .translate(-center.x(), -center.y(), -center.z())
                .scale_uniform(2. / length),
        );
    }

    pub fn transformation(&self) -> Matrix {
        match self {
            Self::Primitive(obj) => obj.transformation,
            Self::Group(_) => Matrix::identity(),
        }
    }

    pub fn transformation_inverse(&self) -> Matrix {
        match self {
            Self::Primitive(obj) => obj.transformation_inverse(),
            Self::Group(_) => Matrix::identity(),
        }
    }

    pub fn bounding_box(&self) -> BoundingBox {
        match self {
            Self::Primitive(obj) => obj.bounding_box(),
            Self::Group(group) => group.bounding_box().clone(),
        }
    }
    pub fn as_group(&self) -> Option<&ObjectGroup> {
        match self {
            Self::Group(group) => Some(group),
            _ => None,
        }
    }
    pub fn as_group_mut(&mut self) -> Option<&mut ObjectGroup> {
        match self {
            Self::Group(group) => Some(group),
            _ => None,
        }
    }
    pub fn as_primitive(&self) -> Option<&PrimitiveObject> {
        match self {
            Self::Primitive(obj) => Some(obj),
            _ => None,
        }
    }
    pub fn as_primitive_mut(&mut self) -> Option<&mut PrimitiveObject> {
        match self {
            Self::Primitive(obj) => Some(obj),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PrimitiveObject {
    shape: Shape,
    material: Material,
    transformation: Matrix,
    transformation_inverse: Matrix,
}

impl PrimitiveObject {
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

    pub fn bounding_box(&self) -> BoundingBox {
        self.shape
            .bounding_box()
            .transform_new(&self.transformation)
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
        self.shape.local_intersect(
            &world_ray.transform_new(&self.transformation_inverse),
            collector,
        );
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

impl Transform for PrimitiveObject {
    fn transform(&mut self, matrix: &Matrix) {
        self.transformation.transform(matrix);
        self.transformation_inverse = self
            .transformation
            .inverse()
            .expect("Object with singular tranfromation matrix cannot be rendered");
    }

    fn transform_new(&self, matrix: &Matrix) -> Self {
        let mut new = self.clone();
        new.transform(matrix);
        new
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn identiy_matrix_is_obj_default_transformation() {
        assert_eq!(
            Object::primitive_with_shape(Shape::Sphere).transformation_inverse(),
            Matrix::identity()
        );
    }

    #[test]
    fn normal_is_normalized() {
        let sphere_obj = Object::primitive_with_shape(Shape::Sphere);

        let frac_sqrt_3_3 = 3_f64.sqrt() / 3.;
        let normal =
            sphere_obj.normal_vector_at(Point::new(frac_sqrt_3_3, frac_sqrt_3_3, frac_sqrt_3_3));
        assert_eq!(normal, normal.normalize());
    }
}
