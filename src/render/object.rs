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

use bounding_box::Bounded;
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
    animations::{Animations, Interpolate},
    intersection::{Intersection, IntersectionCollection, IntersectionCollector},
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

impl Bounded for Object {
    fn bounding_box(&self) -> &BoundingBox {
        match &self.kind {
            ObjectKind::Primitive(p) => p.bounding_box(),
            ObjectKind::Group(group) => group.bounding_box(),
            ObjectKind::Csg(csg) => &csg.bounding_box,
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
        let transform = self.animations.interpolated_with(self, time);
        match &mut self.kind {
            ObjectKind::Primitive(obj) => {
                if transform != Matrix::identity() {
                    obj.transform(&transform);
                }
            }
            ObjectKind::Group(group) => {
                group.animate_with(time, transform);
            }
            ObjectKind::Csg(csg) => csg.animate_with(time, transform),
        }
    }
}

impl From<ObjectKind> for Object {
    fn from(value: ObjectKind) -> Self {
        Self::with_kind(value)
    }
}

impl Object {
    pub fn animated(kind: ObjectKind, animations: Animations) -> Self {
        Self { kind, animations }
    }

    pub fn with_kind(kind: ObjectKind) -> Self {
        Self {
            kind,
            animations: Animations::default(),
        }
    }

    pub fn from_group(group: ObjectGroup) -> Self {
        Self::with_kind(ObjectKind::group(group))
    }
    pub fn from_primitive(obj: PrimitiveObject) -> Self {
        Self::with_kind(ObjectKind::primitive(obj))
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

    pub fn normal_vector_at(&self, scene_point: Point) -> Vector {
        self.normal_vector_at_with_intersection(scene_point, None)
    }

    /// turns this object into a group that contains this object and it's bounding box
    pub fn into_group_with_bbox(&mut self, material: Material) {
        let bbox = self.bounding_box().as_object(material);
        // we turn this object into a group, to preserve object fields like animations
        let kind = std::mem::replace(&mut self.kind, ObjectKind::Group(ObjectGroup::empty()));
        let this = Object::with_kind(kind);

        let group = self.as_group_mut().unwrap();

        group.add_child(this);
        group.add_child(bbox);
    }

    pub fn normal_vector_at_with_intersection<'a>(
        &self,
        scene_point: Point,
        i: Option<&'a Intersection<'a>>,
    ) -> Vector {
        match &self.kind {
            ObjectKind::Primitive(obj) => obj.normal_vector_at_with_intersection(scene_point, i),
            ObjectKind::Group(_) => unreachable!(),
            ObjectKind::Csg(_) => unreachable!(),
        }
    }

    pub fn intersect<'a>(&'a self, scene_ray: &Ray, collector: &mut IntersectionCollector<'a>) {
        match &self.kind {
            ObjectKind::Primitive(obj) => {
                collector.set_next_object(self);
                obj.intersect(scene_ray, collector);
            }
            ObjectKind::Group(group) => group.intersect(scene_ray, collector),
            ObjectKind::Csg(csg) => csg.intersect(scene_ray, collector),
        }
    }

    pub fn intersect_to_collection(&self, scene_ray: &Ray) -> IntersectionCollection {
        let mut collector = IntersectionCollector::new(Some(self), false);
        self.intersect(scene_ray, &mut collector);
        IntersectionCollection::from_collector(scene_ray.clone(), collector)
    }

    pub fn intersect_to_sorted_vec_testing<'a>(&'a self, scene_ray: &Ray) -> Vec<Intersection<'a>> {
        self.intersect_to_collection(scene_ray).into_vec()
    }

    pub fn intersection_times_testing(&self, scene_ray: &Ray) -> Vec<f64> {
        self.intersect_to_sorted_vec_testing(scene_ray)
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
            ObjectKind::Csg(csg) => csg.set_material(material),
        }
    }

    pub fn material_unwrapped(&self) -> &Material {
        self.material().expect("Object has no material")
    }

    pub fn primitive_count(&self) -> usize {
        match &self.kind {
            ObjectKind::Primitive(_) => 1,
            ObjectKind::Group(group) => group.primitive_count(),
            ObjectKind::Csg(_) => 1, // it's a primitive in this context
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
            ObjectKind::Csg(_) => unreachable!(),
        }
    }

    pub fn transformation_inverse(&self) -> Matrix {
        match &self.kind {
            ObjectKind::Primitive(obj) => obj.transformation_inverse(),
            ObjectKind::Group(_) => Matrix::identity(),
            ObjectKind::Csg(_) => unreachable!(),
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

impl Bounded for PrimitiveObject {
    fn bounding_box(&self) -> &BoundingBox {
        &self.bbox
    }
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
    pub fn normal_vector_at(&self, scene_point: Point) -> Vector {
        self.normal_vector_at_with_intersection(scene_point, None)
    }
    pub fn normal_vector_at_with_intersection<'a>(
        &self,
        scene_point: Point,
        i: Option<&'a Intersection<'a>>,
    ) -> Vector {
        let scene_normal = match self.transformation_inverse {
            None => self.shape.local_normal_at(scene_point, i),
            Some(t_inverse) => {
                let object_point = t_inverse * scene_point;

                let object_normal = self.shape.local_normal_at(object_point, i);
                t_inverse.mul_transposed(object_normal)
            }
        };
        scene_normal.normalize()
    }

    pub fn intersect<'a>(&'a self, scene_ray: &Ray, collector: &mut IntersectionCollector<'a>) {
        match self.transformation_inverse {
            None => self.shape.local_intersect(scene_ray, collector),
            Some(transformation_inv) => self
                .shape
                .local_intersect(&scene_ray.transform_new(&transformation_inv), collector),
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
            (Shape::Bbox, _) => {
                self.transformation_inverse =
                    Some(self.bbox.as_cube_transformation().inverse().unwrap());
            }
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

    use std::f64;

    use crate::{
        assert_approx_eq_low_prec,
        primitive::matrix::LocalTransformations,
        render::{
            animations::{Animation, TransformAnimation},
            color::Color,
        },
    };

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

    #[test]
    fn bbox_primitive_obj() {
        let mut expected = Shape::Cube.bounding_box();
        let mut bbox_obj = Shape::Cube
            .bounding_box()
            .as_object(BoundingBox::DEFAULT_DEBUG_BBOX_MATERIAL);

        assert_eq!(&expected, bbox_obj.bounding_box());

        let transformation = Matrix::translation(1., 2., 3.)
            .rotate_y(f64::consts::FRAC_PI_4)
            .transformed();

        expected.transform(&transformation);
        bbox_obj.transform(&transformation);

        assert_eq!(bbox_obj.bounding_box(), &expected);
    }

    #[test]
    fn object_into_group_with_bbox() {
        let material = Material::with_color(Color::red());
        let mut cube = Object::primitive_with_shape(Shape::Cube);

        let transformation = Matrix::rotation_x(f64::consts::FRAC_PI_4)
            .translate(1., 2., 3.)
            .transformed();

        cube.transform(&transformation);
        let expected_bbox = cube.bounding_box().clone();

        cube.into_group_with_bbox(material);

        let group = cube.as_group().unwrap();
        assert_eq!(group.bounding_box(), &expected_bbox);
        assert_eq!(group.children().len(), 2);
        assert_eq!(group.children()[0].bounding_box(), &expected_bbox);
        assert_eq!(group.children()[1].bounding_box(), &expected_bbox);
    }

    #[test]
    fn bbox_primitive_obj_visibility() {
        let material = Material::with_color(Color::red());
        let mut cube = Object::primitive_with_shape(Shape::Cube);

        let transformation = Matrix::rotation_x(f64::consts::FRAC_PI_4)
            .translate(1., 2., 3.)
            .transformed();

        cube.transform(&transformation);

        cube.into_group_with_bbox(material);
        let group_with_bbox = cube;

        let ray = Ray::new(Point::zero(), Vector::new(1., 2., 3.));

        let c = group_with_bbox.intersect_to_collection(&ray);
        let intersected_obj = c.hit().unwrap().object();

        assert_eq!(
            *intersected_obj.as_primitive().unwrap().shape(),
            Shape::Bbox
        );
    }

    #[test]
    fn into_group_with_bbox_preserves_fields() {
        let animations = Animations::with_vec(vec![TransformAnimation::new(
            Animation::default(),
            LocalTransformations::default(),
        )]);

        let kind = ObjectKind::primitive(PrimitiveObject::with_shape(Shape::Cube));
        let mut animated = Object::animated(kind, animations.clone());

        assert_eq!(animated.animations, animations);
        animated.into_group_with_bbox(Material::default());
        assert_eq!(animated.animations, animations);
    }
}
