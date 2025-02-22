use super::{bounding_box::BoundingBox, Object};
use crate::{
    approx_eq::ApproxEq,
    primitive::{
        matrix::{Matrix, Transform},
        point::Point,
        tuple::Tuple,
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
    pub const PARTITION_THRESHOLD: usize = 2;
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
    pub fn partition(&mut self) {
        println!("Partitioning group with {} children", self.primitive_count);
        if self.primitive_count < Self::PARTITION_THRESHOLD {
            return;
        }

        // Calculate the cost of not partitioning
        let self_cost = self.bounding_box().half_area() * self.children.len() as f64;

        // Determine the best axis, position, and cost for partitioning
        let (axis, pos, cost) = self.determine_partition_axis_pos_cost();

        // Avoid partitioning if the cost is not better
        if self_cost <= cost {
            return;
        }

        // Partition the children into left and right groups
        let old_children = std::mem::take(&mut self.children);
        let mut left_group = ObjectGroup::empty();
        let mut right_group = ObjectGroup::empty();

        for child in old_children {
            let child_box = child.bounding_box();
            let child_pos = match axis {
                'x' => child_box.center().x(),
                'y' => child_box.center().y(),
                'z' => child_box.center().z(),
                _ => unreachable!(),
            };

            if child_pos < pos {
                left_group.add_child(child);
            } else {
                right_group.add_child(child);
            }
        }

        // Recursively partition the left and right groups if they are not empty
        if !left_group.children.is_empty() {
            left_group.partition();
            self.children.push(Object::Group(left_group));
        }
        if !right_group.children.is_empty() {
            right_group.partition();
            self.children.push(Object::Group(right_group));
        }
    }

    pub fn determine_partition_axis_pos_cost(&self) -> (char, f64, f64) {
        let axes = ['x', 'y', 'z'];
        let mut best_axis = 'x';
        let mut best_pos = 0.0;
        let mut best_cost = f64::INFINITY;

        // Precompute bounding boxes for all children
        let child_boxes: Vec<_> = self
            .children
            .iter()
            .map(|child| child.bounding_box())
            .collect();

        for &axis in axes.iter() {
            // Sort children along the current axis
            let mut sorted_children: Vec<_> = self
                .children
                .iter()
                .enumerate()
                .map(|(i, child)| {
                    let center = match axis {
                        'x' => child_boxes[i].center().x(),
                        'y' => child_boxes[i].center().y(),
                        'z' => child_boxes[i].center().z(),
                        _ => unreachable!(),
                    };
                    (center, i)
                })
                .collect();
            sorted_children.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Evaluate SAH for each possible split position
            for i in 1..sorted_children.len() {
                let pos = sorted_children[i].0;
                let cost = self.evaluate_sah(axis, pos, &child_boxes);
                if cost < best_cost {
                    best_axis = axis;
                    best_pos = pos;
                    best_cost = cost;
                }
            }
        }

        (best_axis, best_pos, best_cost)
    }

    pub fn evaluate_sah(&self, axis: char, pos: f64, child_boxes: &[BoundingBox]) -> f64 {
        let mut left_box = BoundingBox::empty();
        let mut right_box = BoundingBox::empty();
        let mut left_count = 0;
        let mut right_count = 0;

        for child_box in child_boxes.iter() {
            let box_pos = match axis {
                'x' => child_box.center().x(),
                'y' => child_box.center().y(),
                'z' => child_box.center().z(),
                _ => unreachable!(),
            };
            if box_pos < pos {
                left_box.add_bounding_box(child_box);
                left_count += 1;
            } else {
                right_box.add_bounding_box(child_box);
                right_count += 1;
            }
        }

        // If one of the groups is empty, return infinity (invalid split)
        if left_count == 0 || right_count == 0 {
            return f64::INFINITY;
        }

        // Calculate SAH cost
        let left_cost = left_box.half_area() * left_count as f64;
        let right_cost = right_box.half_area() * right_count as f64;
        let total_cost = left_cost + right_cost;

        // Debug prints
        println!(
        "Axis: {}, Pos: {}, Left Count: {}, Right Count: {}, Left Cost: {}, Right Cost: {}, Total Cost: {}",
        axis, pos, left_count, right_count, left_cost, right_cost, total_cost
    );

        total_cost
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
