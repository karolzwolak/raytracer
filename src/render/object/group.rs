use super::{bounding_box::BoundingBox, Object};
use crate::{
    primitive::{
        matrix::{Matrix, Transform},
        point::Point,
    },
    render::{
        intersection::IntersectionCollector,
        material::{self, Material},
        ray::Ray,
    },
};

#[derive(Clone, Debug, PartialEq)]
/// A group of objects that can be transformed simultaneously.
/// However, children added later will not be affected by previous transformations.
/// It also features automatic bounding_box calculation, that reduce ray intersection checks.
pub struct ObjectGroup {
    children: Vec<Object>,
    bounding_box: BoundingBox,
    primitive_count: usize,
}

impl ObjectGroup {
    pub const PARTITION_THRESHOLD: usize = 8;
    const BBOX_SPLIT_POWER: usize = 6;
    const DIVISION_BBOX_LEN_FACTOR: f64 = 5.0e-3;

    fn with_bounding_box(children: Vec<Object>, bounding_box: BoundingBox) -> Self {
        let count = children
            .iter()
            .fold(0, |acc, child| acc + child.primitive_count());
        Self {
            children,
            bounding_box,
            primitive_count: count,
        }
    }

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
            primitive_count: 0,
        }
    }

    /// Recursively applies the transformation to all children.
    pub fn set_material(&mut self, material: Material) {
        for child in self.children.iter_mut() {
            child.set_material(material.clone());
        }
    }
    pub fn add_child(&mut self, child: Object) {
        self.bounding_box.add_bounding_box(&child.bounding_box());
        self.primitive_count += child.primitive_count();
        self.children.push(child);
    }
    pub fn add_children(&mut self, children: impl IntoIterator<Item = Object>) {
        for child in children {
            self.add_child(child);
        }
    }
    fn divide(&mut self) -> (Vec<BoundingBox>, Vec<Vec<Object>>) {
        let mut boxes = self.bounding_box().split_n(Self::BBOX_SPLIT_POWER);
        if boxes.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let box_len = (boxes[0].max - boxes[0].min).magnitude();
        let min_dist_to_be_divided = box_len * Self::DIVISION_BBOX_LEN_FACTOR;
        let mut vectors = vec![vec![]; boxes.len()];

        std::mem::take(&mut self.children)
            .into_iter()
            .for_each(|child| {
                let mut child_box = child.bounding_box();
                child_box.limit_dimensions();

                let mut min_id = 0;
                let mut min_d = f64::INFINITY;

                let dist_to_group = self.bounding_box.distance(&child_box);
                if dist_to_group <= min_dist_to_be_divided {
                    self.children.push(child);
                    return;
                }

                for (id, b) in boxes.iter().enumerate() {
                    let d = b.distance(&child_box);
                    if d < min_d {
                        min_id = id;
                        min_d = d;
                    }
                }

                boxes[min_id].add_bounding_box(&child_box);
                vectors[min_id].push(child);
            });
        (boxes, vectors)
    }
    fn partition_iter(root: &mut ObjectGroup) {
        let mut group_stack = vec![root];

        while let Some(group) = group_stack.pop() {
            group.bounding_box.limit_dimensions();
            if group.primitive_count < Self::PARTITION_THRESHOLD {
                continue;
            }

            let (boxes, vectors) = group.divide();

            group.children.extend(
                vectors
                    .into_iter()
                    .zip(boxes.into_iter())
                    .filter(|(v, _)| !v.is_empty())
                    .map(|(children, bounding_box)| {
                        ObjectGroup::with_bounding_box(children, bounding_box).into()
                    }),
            );

            group.children.sort_unstable_by(|a, b| {
                let p = Point::zero();
                let time_a = a.bounding_box().intersection_time_from_point(p);
                let time_b = b.bounding_box().intersection_time_from_point(p);
                time_a.partial_cmp(&time_b).unwrap()
            });

            group_stack.extend(
                group
                    .children
                    .iter_mut()
                    .filter_map(|child| child.as_group_mut().map(|g| g as &mut ObjectGroup)),
            );
        }
    }
    pub fn partition(&mut self) {
        println!(
            "Partitioning group with {} primitives",
            self.primitive_count
        );
        Self::partition_iter(self);
    }
    pub fn into_children(self) -> Vec<Object> {
        self.children
    }
    fn intersect_iter<'a>(
        root: &'a ObjectGroup,
        world_ray: &Ray,
        collector: &mut IntersectionCollector<'a>,
    ) {
        let mut stack = Vec::new();
        match root.bounding_box.intersection_time(world_ray) {
            None => return,
            Some(t) => stack.push((root, t)),
        }
        while let Some((group, time)) = stack.pop() {
            if time >= collector.hit_time() {
                continue;
            }
            stack.extend(group.children.iter().rev().filter_map(|child| match child {
                Object::Group(g) => g.bounding_box.intersection_time(world_ray).map(|t| (g, t)),
                Object::Primitive(_) => {
                    child.intersect(world_ray, collector);
                    None
                }
            }));
        }
    }
    pub fn intersect<'a>(&'a self, world_ray: &Ray, collector: &mut IntersectionCollector<'a>) {
        Self::intersect_iter(self, world_ray, collector)
    }

    pub fn bounding_box(&self) -> &BoundingBox {
        &self.bounding_box
    }

    pub fn children(&self) -> &[Object] {
        self.children.as_ref()
    }
    pub fn children_mut(&mut self) -> &mut [Object] {
        self.children.as_mut()
    }
    pub fn add_bounding_box_as_obj(&mut self) {
        self.children.push(self.bounding_box.as_object())
    }

    pub fn primitive_count(&self) -> usize {
        self.primitive_count
    }
}

impl Default for ObjectGroup {
    fn default() -> Self {
        Self::empty()
    }
}

impl Transform for ObjectGroup {
    fn transform(&mut self, matrix: &Matrix) {
        for child in self.children.iter_mut() {
            child.transform(matrix);
        }
        self.bounding_box.transform(matrix);
    }

    fn transform_new(&self, matrix: &Matrix) -> Self {
        let mut new = self.clone();
        new.transform(matrix);
        new
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        approx_eq::ApproxEq,
        assert_approx_eq_low_prec,
        primitive::{matrix::Matrix, point::Point, tuple::Tuple, vector::Vector},
        render::{
            intersection::IntersectionCollection,
            object::{group::ObjectGroup, shape::Shape, Object, PrimitiveObject},
            ray::Ray,
        },
    };

    #[test]
    fn intersecting_ray_with_empty_group() {
        let object = Object::group_with_children(Vec::new());
        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        assert!(object.intersect_to_sorted_vec_testing(&ray).is_empty());
    }

    #[test]
    fn intersecting_ray_with_nonempty_group() {
        let s1 = Object::primitive_with_shape(Shape::Sphere);
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
            Object::primitive_with_transformation(Shape::Sphere, Matrix::translation(5., 0., 0.));
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
}
