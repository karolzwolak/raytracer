pub mod bounding_box;
pub mod cone;
pub mod csg;
pub mod cube;
pub mod cylinder;
pub mod group;
pub mod plane;
pub mod shape;
pub mod smooth_triangle;
pub mod sphere;
pub mod triangle;

use csg::CsgObject;

use crate::{
    approx_eq::ApproxEq,
    primitive::{
        matrix::{Matrix, Transform},
        point::Point,
        tuple::Tuple,
        vector::Vector,
    },
};

use self::{bounding_box::BoundingBox, group::ObjectGroup, shape::Shape};

use super::{
    animations::Animations,
    intersection::{Intersection, IntersectionCollector},
    material::Material,
    ray::Ray,
};

#[derive(Clone, Debug, PartialEq)]
pub enum ObjectKind {
    Primitive(Box<PrimitiveObject>),
    Group(ObjectGroup),
    Csg(Box<CsgObject>),
}

impl ObjectKind {
    pub fn group(group: ObjectGroup) -> Self {
        Self::Group(group)
    }
    pub fn primitive(obj: PrimitiveObject) -> Self {
        Self::Primitive(Box::new(obj))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Object {
    kind: ObjectKind,
    animations: Animations,
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
        match &mut self.kind {
            ObjectKind::Primitive(obj) => obj.transform(matrix),
            ObjectKind::Group(group) => group.transform(matrix),
            ObjectKind::Csg(csg) => csg.transform(matrix),
        }
    }
}

impl Object {
    pub fn includes(&self, other: &Object) -> bool {
        self == other
            || match &self.kind {
                ObjectKind::Group(group) => group.includes(other),
                _ => false,
            }
    }
}

impl Object {
    pub fn animate(&mut self, time: f64) {
        let transform = self.animations.matrix_at(time);
        match &mut self.kind {
            ObjectKind::Primitive(obj) => {
                if transform != Matrix::identity() {
                    obj.transform(&transform);
                }
            }
            ObjectKind::Group(group) => {
                group.animate_with(time, transform);
            }
            ObjectKind::Csg(_) => todo!(),
        }
    }
}

impl Object {
    pub fn animated(kind: ObjectKind, animations: Animations) -> Self {
        Self { kind, animations }
    }
    pub fn from_group(group: ObjectGroup) -> Self {
        Self {
            animations: Animations::empty(),
            kind: ObjectKind::Group(group),
        }
    }
    pub fn from_primitive(obj: PrimitiveObject) -> Self {
        Self {
            animations: Animations::empty(),
            kind: ObjectKind::Primitive(Box::new(obj)),
        }
    }
    pub fn group_with_children(children: Vec<Object>) -> Self {
        Self::from_group(ObjectGroup::new(children))
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
        match &self.kind {
            ObjectKind::Primitive(obj) => obj.normal_vector_at_with_intersection(world_point, i),
            ObjectKind::Group(_) => todo!(),
            ObjectKind::Csg(_) => todo!(),
        }
    }

    pub fn intersect<'a>(&'a self, world_ray: &Ray, collector: &mut IntersectionCollector<'a>) {
        match &self.kind {
            ObjectKind::Primitive(obj) => {
                collector.set_next_object(self);
                obj.intersect(world_ray, collector);
            }
            ObjectKind::Group(group) => group.intersect(world_ray, collector),
            ObjectKind::Csg(_) => todo!(),
        }
    }

    pub fn intersect_to_sorted_vec_testing<'a>(&'a self, world_ray: &Ray) -> Vec<Intersection<'a>> {
        let mut collector = IntersectionCollector::new(Some(self), false);
        self.intersect(world_ray, &mut collector);
        collector.collect_sorted()
    }

    pub fn intersection_times_testing(&self, world_ray: &Ray) -> Vec<f64> {
        self.intersect_to_sorted_vec_testing(world_ray)
            .iter_mut()
            .map(|i| i.time())
            .collect()
    }

    pub fn is_intersected_by_ray(&self, ray: &Ray) -> bool {
        !self.intersect_to_sorted_vec_testing(ray).is_empty()
    }

    pub fn material(&self) -> Option<&Material> {
        self.as_primitive().map(|p| p.material())
    }

    pub fn material_mut(&mut self) -> Option<&mut Material> {
        self.as_primitive_mut().map(|p| p.material_mut())
    }

    pub fn set_material(&mut self, material: Material) {
        match &mut self.kind {
            ObjectKind::Primitive(obj) => obj.set_material(material),
            ObjectKind::Group(group) => group.set_material(material),
            ObjectKind::Csg(_) => todo!(),
        }
    }

    pub fn material_unwrapped(&self) -> &Material {
        self.material().expect("Object has no material")
    }

    pub fn primitive_count(&self) -> usize {
        match &self.kind {
            ObjectKind::Primitive(_) => 1,
            ObjectKind::Group(group) => group.primitive_count(),
            ObjectKind::Csg(_) => todo!(),
        }
    }

    pub fn normalize_to_longest_dim(&mut self) {
        let bb = self.bounding_box();
        let diff = bb.max - bb.min;
        let length = diff.x().max(diff.y()).max(diff.z());

        self.transform(&Matrix::scaling_uniform(2. / length));
    }

    pub fn center(&mut self) {
        let bb = self.bounding_box();
        let center = bb.center();

        self.transform(&Matrix::translation(-center.x(), -center.y(), -center.z()));
    }

    pub fn center_above_oy(&mut self) {
        let bb = self.bounding_box();
        let center = bb.center();
        self.transform(&Matrix::translation(-center.x(), -bb.min.y(), -center.z()));
    }

    pub fn transformation(&self) -> Matrix {
        match &self.kind {
            ObjectKind::Primitive(obj) => obj
                .transformation_inverse
                .unwrap_or_default()
                .inverse()
                .unwrap(),
            ObjectKind::Group(_) => Matrix::identity(),
            ObjectKind::Csg(_) => todo!(),
        }
    }

    pub fn transformation_inverse(&self) -> Matrix {
        match &self.kind {
            ObjectKind::Primitive(obj) => obj.transformation_inverse(),
            ObjectKind::Group(_) => Matrix::identity(),
            ObjectKind::Csg(_) => todo!(),
        }
    }

    pub fn bounding_box(&self) -> BoundingBox {
        match &self.kind {
            ObjectKind::Primitive(obj) => obj.bounding_box(),
            ObjectKind::Group(group) => group.bounding_box().clone(),
            ObjectKind::Csg(_) => todo!(),
        }
    }
    pub fn as_group(&self) -> Option<&ObjectGroup> {
        match &self.kind {
            ObjectKind::Group(group) => Some(group),
            _ => None,
        }
    }
    pub fn as_group_mut(&mut self) -> Option<&mut ObjectGroup> {
        match &mut self.kind {
            ObjectKind::Group(group) => Some(group),
            _ => None,
        }
    }
    pub fn as_primitive(&self) -> Option<&PrimitiveObject> {
        match &self.kind {
            ObjectKind::Primitive(obj) => Some(obj),
            _ => None,
        }
    }
    pub fn as_primitive_mut(&mut self) -> Option<&mut PrimitiveObject> {
        match &mut self.kind {
            ObjectKind::Primitive(obj) => Some(obj),
            _ => None,
        }
    }

    pub fn kind(&self) -> &ObjectKind {
        &self.kind
    }

    pub fn animations(&self) -> &Animations {
        &self.animations
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrimitiveObject {
    shape: Shape,
    material: Material,
    transformation_inverse: Option<Matrix>,
    bbox: BoundingBox,
}

impl PrimitiveObject {
    pub fn new(shape: Shape, material: Material, transformation: Matrix) -> Self {
        let mut res = PrimitiveObject {
            bbox: shape.bounding_box(),
            shape,
            material,
            transformation_inverse: None,
        };
        if !transformation.approx_eq(&Matrix::identity()) {
            res.transform(&transformation);
        }
        res
    }

    pub fn bounding_box(&self) -> BoundingBox {
        self.bbox.clone()
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
    pub fn center(&self) -> Point {
        self.bbox.center()
    }
    pub fn transformation_inverse(&self) -> Matrix {
        self.transformation_inverse.unwrap_or_default()
    }
    pub fn normal_vector_at(&self, world_point: Point) -> Vector {
        self.normal_vector_at_with_intersection(world_point, None)
    }
    pub fn normal_vector_at_with_intersection<'a>(
        &self,
        world_point: Point,
        i: Option<&'a Intersection<'a>>,
    ) -> Vector {
        let world_normal = match self.transformation_inverse {
            None => self.shape.local_normal_at(world_point, i),
            Some(t_inverse) => {
                let object_point = t_inverse * world_point;

                let object_normal = self.shape.local_normal_at(object_point, i);
                t_inverse.mul_transposed(object_normal)
            }
        };
        world_normal.normalize()
    }

    pub fn intersect<'a>(&'a self, world_ray: &Ray, collector: &mut IntersectionCollector<'a>) {
        match self.transformation_inverse {
            None => self.shape.local_intersect(world_ray, collector),
            Some(transformation_inv) => self
                .shape
                .local_intersect(&world_ray.transform_new(&transformation_inv), collector),
        };
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
        self.bbox.transform(matrix);
        match (&mut self.shape, &mut self.transformation_inverse) {
            (Shape::Triangle(t), _) => t.transform(matrix),
            (Shape::SmoothTriangle(t), _) => t.transform(matrix),
            (_, Some(inv)) => {
                *inv = matrix.inverse().unwrap().transform_new(inv);
            }
            (_, None) => {
                self.transformation_inverse = Some(matrix.inverse().unwrap());
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::assert_approx_eq_low_prec;

    use super::*;

    #[test]
    fn identiy_matrix_is_obj_default_transformation() {
        assert_approx_eq_low_prec!(
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
        assert_approx_eq_low_prec!(normal, normal.normalize());
    }
}
