use crate::Material;
use crate::ObjectKind;
use crate::{
    core::{
        matrix::{Matrix, Transform},
        tuple::Axis,
    },
    render::{intersection::IntersectionCollector, ray::Ray},
    Bounded, BoundingBox,
};

use super::Object;

#[derive(Clone, Debug, PartialEq, Default)]
/// A group of objects that can be transformed simultaneously.
/// However, children added later will not be affected by previous transformations.
/// It also features automatic bounding_box calculation, that reduce ray intersection checks.
pub struct ObjectGroup {
    children: Vec<Object>,
    bounding_box: BoundingBox,
    core_count: usize,
}

impl Bounded for ObjectGroup {
    fn bounding_box(&self) -> &BoundingBox {
        &self.bounding_box
    }
}

impl ObjectGroup {
    const CANDIDATE_POS_NUMBER: usize = 5;

    pub fn new(children: Vec<Object>) -> Self {
        let mut res = Self::empty();
        for child in children {
            res.add_child(child);
        }
        res
    }

    pub fn with_transformations(children: Vec<Object>, transformation: Matrix) -> Self {
        let mut group = Self::new(children);
        group.transform(&transformation);
        group
    }

    pub fn with_transformations_and_material(
        children: Vec<Object>,
        transformation: Matrix,
        material: Material,
    ) -> Self {
        let mut group = Self::new(children);
        group.transform(&transformation);
        group.set_material(material);
        group
    }

    pub fn empty() -> Self {
        Self {
            children: Vec::new(),
            bounding_box: BoundingBox::empty(),
            core_count: 0,
        }
    }

    /// Recursively applies the transformation to all children.
    pub fn set_material(&mut self, material: Material) {
        for child in self.children.iter_mut() {
            child.set_material(material.clone());
        }
    }
    pub fn add_child(&mut self, child: Object) {
        self.bounding_box.add_bounding_box(child.bounding_box());
        self.core_count += child.core_count();
        self.children.push(child);
    }
    pub fn add_children(&mut self, children: impl IntoIterator<Item = Object>) {
        for child in children {
            self.add_child(child);
        }
    }

    fn choose_split(&self) -> (Axis, f64, f64) {
        let mut best_axis = Axis::X;
        let mut best_pos = 0.0;
        let mut best_cost = f64::INFINITY;
        let bbox_len_vec = self.bounding_box.length_vec();

        for i in 0..Self::CANDIDATE_POS_NUMBER {
            let coef = (i as f64 + 1.) / (Self::CANDIDATE_POS_NUMBER as f64 + 1.);
            let pos = self.bounding_box.min + bbox_len_vec * coef;
            for axis in Axis::iter() {
                let pos = pos[axis];

                let cost = self.calculate_split_cost(axis, pos);
                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                    best_pos = pos;
                }
            }
        }
        (best_axis, best_pos, best_cost)
    }
    fn calculate_split_cost(&self, axis: Axis, pos: f64) -> f64 {
        let mut left_bbox = BoundingBox::empty();
        let mut left_count = 0;
        let mut right_bbox = BoundingBox::empty();
        let mut right_count = 0;

        for child in self.children.iter() {
            let bbox = child.bounding_box();
            if bbox.center()[axis] < pos {
                left_bbox.add_bounding_box(bbox);
                left_count += child.core_count();
            } else {
                right_bbox.add_bounding_box(bbox);
                right_count += child.core_count();
            }
        }
        if left_count == 0 || right_count == 0 {
            return f64::INFINITY;
        }
        left_bbox.half_area() * left_count as f64 + right_bbox.half_area() * right_count as f64
    }

    fn split(&mut self, axis: Axis, pos: f64) -> (ObjectGroup, ObjectGroup) {
        let mut left = Self::empty();
        let mut right = Self::empty();
        for child in self.children.drain(..) {
            let child_bbox = child.bounding_box();
            if child_bbox.center()[axis] < pos {
                left.add_child(child);
            } else {
                right.add_child(child);
            }
        }
        (left, right)
    }
    pub fn build_bvh(&mut self) {
        if self.children.len() <= 2 {
            for child in self.children.iter_mut() {
                if let Some(group) = child.as_group_mut() {
                    group.build_bvh();
                }
            }
            return;
        }
        let (axis, pos, cost) = self.choose_split();
        if cost >= self.sah_cost() {
            return;
        }
        let (mut left, mut right) = self.split(axis, pos);

        if left.core_count() > 0 {
            left.build_bvh();
            self.children.push(left.into());
        }
        if right.core_count() > 0 {
            right.build_bvh();
            self.children.push(right.into());
        }
    }
    pub fn into_children(self) -> Vec<Object> {
        self.children
    }
    fn intersect_iter<'a>(
        root: &'a ObjectGroup,
        scene_ray: &Ray,
        collector: &mut IntersectionCollector<'a>,
    ) {
        let mut stack = Vec::new();
        match root.bounding_box.intersection_time(scene_ray) {
            None => return,
            Some(t) => stack.push((root, t)),
        }
        while let Some((group, time)) = stack.pop() {
            if time >= collector.hit_time() {
                continue;
            }
            stack.extend(
                group
                    .children
                    .iter()
                    .rev()
                    .filter_map(|child| match child.kind() {
                        ObjectKind::Group(g) => {
                            g.bounding_box.intersection_time(scene_ray).map(|t| (g, t))
                        }
                        ObjectKind::Primitive(_) | ObjectKind::Csg(_) => {
                            child.intersect(scene_ray, collector);
                            None
                        }
                    }),
            );
        }
    }
    pub fn intersect<'a>(&'a self, scene_ray: &Ray, collector: &mut IntersectionCollector<'a>) {
        Self::intersect_iter(self, scene_ray, collector)
    }

    pub fn children(&self) -> &[Object] {
        self.children.as_ref()
    }
    pub fn children_mut(&mut self) -> &mut [Object] {
        self.children.as_mut()
    }
    pub fn add_bounding_box_as_obj(&mut self, material: Material) {
        self.children.push(self.bounding_box.as_object(material))
    }

    pub fn core_count(&self) -> usize {
        self.core_count
    }

    pub fn sah_cost(&self) -> f64 {
        self.bounding_box.half_area() * self.core_count as f64
    }

    pub fn includes(&self, other: &Object) -> bool {
        self.children.iter().any(|child| child.includes(other))
    }
}

impl ObjectGroup {
    fn recalculate_bbox(&mut self) {
        self.bounding_box = BoundingBox::empty();
        for child in self.children.iter() {
            self.bounding_box.add_bounding_box(child.bounding_box());
        }
    }
    pub fn animate(&mut self, time: f64) {
        for child in self.children.iter_mut() {
            child.animate(time);
        }
        self.recalculate_bbox();
    }
    pub fn animate_with(&mut self, time: f64, transform: Matrix) {
        if transform != Matrix::identity() {
            self.transform(&transform);
        }
        self.animate(time);
    }
}

impl Transform for ObjectGroup {
    fn transform(&mut self, matrix: &Matrix) {
        for child in self.children.iter_mut() {
            child.transform(matrix);
        }
        self.bounding_box.transform(matrix);
    }
}

#[cfg(test)]
mod tests {
    use super::ObjectGroup;
    use crate::{
        approx_eq::ApproxEq,
        assert_approx_eq_low_prec,
        core::{
            matrix::{LocalTransformations, Matrix, Transformation},
            point::Point,
            tuple::Tuple,
            vector::Vector,
        },
        render::{intersection::IntersectionCollection, ray::Ray},
        scene::animation::{
            Animation, AnimationDirection, AnimationRepeat, AnimationTiming, Animations,
            TransformAnimation,
        },
        Object, PrimitiveObject, Shape,
    };
    use crate::{Bounded, ObjectKind};

    #[test]
    fn intersecting_ray_with_empty_group() {
        let object = Object::group_with_children(Vec::new());
        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        assert!(object.intersect_to_sorted_vec_testing(&ray).is_empty());
    }

    #[test]
    fn intersecting_ray_with_nonempty_group() {
        let s1 = Object::core_with_shape(Shape::Sphere);
        let s2 = PrimitiveObject::sphere(Point::new(0., 0., -3.), 1.).into();
        let s3 = PrimitiveObject::sphere(Point::new(5., 0., 0.), 1.).into();

        let object = Object::group_with_children(vec![s1, s2, s3]);

        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let mut xs = IntersectionCollection::from_ray_and_obj_testing(ray, &object);
        let data = xs.vec_sorted();
        let group = object.as_group().unwrap();

        assert_eq!(data.len(), 4);

        assert!(std::ptr::eq(data[0].object(), &group.children[1]));
        assert!(std::ptr::eq(data[1].object(), &group.children[1]));
        assert!(std::ptr::eq(data[2].object(), &group.children[0]));
        assert!(std::ptr::eq(data[3].object(), &group.children[0]));
    }

    #[test]
    fn intersecting_transformed_group() {
        let sphere = PrimitiveObject::sphere(Point::new(5., 0., 0.), 1.).into();
        let object = Object::from_group(ObjectGroup::with_transformations(
            vec![sphere],
            Matrix::scaling_uniform(2.),
        ));

        let ray = Ray::new(Point::new(10., 0., -10.), Vector::new(0., 0., 1.));
        assert_eq!(object.intersect_to_sorted_vec_testing(&ray).len(), 2);
    }

    #[test]
    fn normal_on_group_child() {
        let sphere =
            Object::core_with_transformation(Shape::Sphere, Matrix::translation(5., 0., 0.));
        let g2 =
            ObjectGroup::with_transformations(vec![sphere], Matrix::scaling(1., 2., 3.)).into();
        let g1: Object = ObjectGroup::with_transformations(
            vec![g2],
            Matrix::rotation_y(std::f64::consts::FRAC_PI_2),
        )
        .into();

        let sphere = &g1.as_group().unwrap().children[0]
            .as_group()
            .unwrap()
            .children[0];
        let normal = sphere.normal_vector_at(Point::new(1.7321, 1.1547, -5.5774));
        assert_approx_eq_low_prec!(normal, Vector::new(0.2857, 0.4286, -0.8571));
    }

    #[test]
    fn animating_children_updates_bbox() {
        let sphere = PrimitiveObject::sphere(Point::zero(), 1.);
        let translate = Transformation::Translation(5., 2., 10.);
        let transfom = LocalTransformations::from(vec![translate]);
        let animation = Animation::new(
            0.,
            1.,
            AnimationDirection::Normal,
            AnimationTiming::Linear,
            AnimationRepeat::Infinite,
        );
        let mut animated_object = Object::animated(
            ObjectKind::core(sphere.clone()),
            Animations::from(vec![TransformAnimation::new(animation, transfom)]),
        );

        let mut group = ObjectGroup::new(vec![animated_object.clone()]);

        assert_eq!(group.bounding_box(), animated_object.bounding_box());
        assert_eq!(animated_object.bounding_box(), sphere.bounding_box());
        group.animate(0.5);
        animated_object.animate(0.5);
        assert_eq!(group.bounding_box(), animated_object.bounding_box());
        assert_ne!(animated_object.bounding_box(), sphere.bounding_box());
    }
}
