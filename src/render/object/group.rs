use crate::{
    primitive::matrix::Matrix,
    render::{intersection::IntersectionCollector, material::Material, ray::Ray},
};

use super::{bounding_box::BoundingBox, shape::Shape, Object};

#[derive(Clone, Debug)]
/// A group of objects that can be transformed simultaneously.
/// However, children added later will not be affected by previous transformations.
/// It also features automatic bounding_box calculation, that reduce ray intersection checks.
pub struct ObjectGroup {
    children: Vec<Object>,
    bounding_box: BoundingBox,
}

impl ObjectGroup {
    pub const PARTITION_THRESHOLD: usize = 32;

    fn with_bounding_box(children: Vec<Object>, bounding_box: BoundingBox) -> Self {
        Self {
            children,
            bounding_box,
        }
    }

    pub fn new(children: Vec<Object>) -> Self {
        let mut bounding_box = BoundingBox::empty();
        for child in children.iter() {
            bounding_box.add_bounding_box(child.bounding_box());
        }
        Self {
            children,
            bounding_box,
        }
    }
    pub fn with_transformations(children: Vec<Object>, transformation: Matrix) -> Self {
        let mut group = Self::new(children);
        group.apply_transformation(transformation);
        group
    }
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }
    pub fn apply_transformation(&mut self, matrix: Matrix) {
        for child in self.children.iter_mut() {
            child.apply_transformation(matrix);
        }
        self.bounding_box.transform(matrix);
    }
    pub fn set_material(&mut self, material: Material) {
        for child in self.children.iter_mut() {
            child.set_material(material.clone());
        }
    }
    pub fn add_child(&mut self, child: Object) {
        self.bounding_box.add_bounding_box(child.bounding_box());
        self.children.push(child);
    }
    pub fn add_children(&mut self, children: impl IntoIterator<Item = Object>) {
        for child in children {
            self.add_child(child);
        }
    }
    pub fn partition(&mut self) {
        if self.children.len() < Self::PARTITION_THRESHOLD {
            return;
        }
        let old_children = std::mem::take(&mut self.children);
        let (left_box, right_box) = self.bounding_box.split_along_longest_axis();

        let mut left_group = ObjectGroup::with_bounding_box(Vec::new(), left_box);
        let mut right_group = ObjectGroup::with_bounding_box(Vec::new(), right_box);

        for child in old_children {
            let child_box = child.bounding_box();
            let left_dist = left_group.bounding_box().distance(&child_box);
            let right_dist = right_group.bounding_box().distance(&child_box);

            if left_dist < right_dist {
                left_group.add_child(child);
            } else {
                right_group.add_child(child);
            }
        }

        if !left_group.children.is_empty() {
            left_group.partition();
            self.children.push(left_group.into_object());
        }
        if !right_group.children.is_empty() {
            right_group.partition();
            self.children.push(right_group.into_object());
        }
    }
    pub fn into_children(self) -> Vec<Object> {
        self.children
    }
    pub fn into_shape(self) -> Shape {
        Shape::Group(self)
    }
    pub fn into_object(self) -> Object {
        Object::with_shape(self.into_shape())
    }
    pub fn intersect<'a>(&'a self, world_ray: &Ray, collector: &mut IntersectionCollector<'a>) {
        if !self.bounding_box.is_intersected(world_ray) {
            return;
        }
        for child in self.children.iter() {
            child.intersect_with_collector(world_ray, collector)
        }
    }
    pub fn bounding_box(&self) -> &BoundingBox {
        &self.bounding_box
    }

    pub fn children(&self) -> &[Object] {
        self.children.as_ref()
    }
    pub fn add_bounding_box_as_obj(&mut self) {
        self.children.push(self.bounding_box.as_object())
    }
}

impl Default for ObjectGroup {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        approx_eq::ApproxEq,
        primitive::{matrix::Matrix, point::Point, tuple::Tuple, vector::Vector},
        render::{
            intersection::IntersectionCollection,
            object::{group::ObjectGroup, shape::Shape, Object},
            ray::Ray,
        },
    };

    #[test]
    fn intersecting_ray_with_empty_group() {
        let group = ObjectGroup::empty();
        let object = Object::with_shape(group.into_shape());
        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        assert!(object.intersect_to_vec(&ray).is_empty());
    }

    #[test]
    fn intersecting_ray_with_nonempty_group() {
        let s1 = Object::with_shape(Shape::Sphere);
        let s2 = Object::sphere(Point::new(0., 0., -3.), 1.);
        let s3 = Object::sphere(Point::new(5., 0., 0.), 1.);

        let group = ObjectGroup::new(vec![s1, s2, s3]);
        let object = Object::with_shape(group.into_shape());

        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let xs = IntersectionCollection::from_ray_and_obj(ray, &object);
        let data = xs.vec();

        match object.shape() {
            Shape::Group(group) => {
                assert_eq!(data.len(), 4);

                assert!(std::ptr::eq(data[0].object(), &group.children[1]));
                assert!(std::ptr::eq(data[1].object(), &group.children[1]));
                assert!(std::ptr::eq(data[2].object(), &group.children[0]));
                assert!(std::ptr::eq(data[3].object(), &group.children[0]));
            }
            _ => panic!("expected Shape::Group"),
        }
    }

    #[test]
    fn intersecting_transformed_group() {
        let sphere = Object::sphere(Point::new(5., 0., 0.), 1.);
        let group = ObjectGroup::with_transformations(vec![sphere], Matrix::scaling_uniform(2.));
        let object = Object::with_shape(group.into_shape());

        let ray = Ray::new(Point::new(10., 0., -10.), Vector::new(0., 0., 1.));
        assert_eq!(object.intersect_to_vec(&ray).len(), 2);
    }

    #[test]
    fn normal_on_group_child() {
        let sphere = Object::with_transformation(Shape::Sphere, Matrix::translation(5., 0., 0.));
        let g2 = Object::group(vec![sphere], Matrix::scaling(1., 2., 3.));
        let g1 = Object::group(vec![g2], Matrix::rotation_y(std::f64::consts::FRAC_PI_2));

        let sphere = &g1.get_group().unwrap().children[0]
            .get_group()
            .unwrap()
            .children[0];
        let normal = sphere.normal_vector_at(Point::new(1.7321, 1.1547, -5.5774));
        assert!(normal.approx_eq_low_prec(&Vector::new(0.2857, 0.4286, -0.8571)));
    }
}
