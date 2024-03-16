use crate::{
    approx_eq::{self},
    primitive::{point::Point, vector::Vector},
};

use super::{
    material::{Material, AIR_REFRACTIVE_INDEX},
    object::Object,
    ray::Ray,
};

/// Will panic if you try to add intersection without setting next_object to Some first
pub struct IntersectionCollector<'a> {
    vec: Vec<Intersection<'a>>,
    next_object: Option<&'a Object>,
}

impl<'a> IntersectionCollector<'a> {
    pub fn new() -> Self {
        Self {
            vec: Vec::new(),
            next_object: None,
        }
    }
    pub fn with_next_object(next_object: &'a Object) -> Self {
        Self {
            vec: Vec::new(),
            next_object: Some(next_object),
        }
    }
    pub fn set_next_object(&mut self, object: &'a Object) {
        self.next_object = Some(object);
    }
    fn get_next_object_expect(&self) -> &'a Object {
        self.next_object
            .expect("Internal error: tried adding intersection without providing object reference")
    }
    /// Will panic if next_object is None
    pub fn add(&mut self, time: f64) {
        let obj = self.get_next_object_expect();
        self.vec.push(Intersection::new(time, obj));
    }
    /// Will panic if next_object is None
    pub fn add_uv(&mut self, time: f64, u: f64, v: f64) {
        let obj = self.get_next_object_expect();
        self.vec.push(Intersection::new_with_uv(time, obj, u, v));
    }
    pub fn collect_sorted(mut self) -> Vec<Intersection<'a>> {
        self.vec
            .sort_unstable_by(|i1, i2| i1.time().partial_cmp(&i2.time()).unwrap());
        self.vec
    }
}

impl<'a> Default for IntersectionCollector<'a> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
pub struct Intersection<'a> {
    time: f64,
    intersected_object: &'a Object,
    u: f64,
    v: f64,
}

fn same_obj_ref(a: &Object, b: &Object) -> bool {
    std::ptr::eq(a, b)
}

impl<'a> PartialEq for Intersection<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && same_obj_ref(self.object(), other.object())
    }
}

impl<'a> Intersection<'a> {
    pub fn new(time: f64, intersected_object: &'a Object) -> Self {
        Self {
            time,
            intersected_object,
            u: f64::NEG_INFINITY,
            v: f64::NEG_INFINITY,
        }
    }
    pub fn new_with_uv(time: f64, intersected_object: &'a Object, u: f64, v: f64) -> Self {
        Self {
            time,
            intersected_object,
            u,
            v,
        }
    }

    pub fn time(&self) -> f64 {
        self.time
    }
    pub fn object(&self) -> &'a Object {
        self.intersected_object
    }

    pub fn computations(&self, ray: &Ray) -> IntersecComputations {
        IntersecComputations::from_intersection(self, ray)
    }

    pub fn u(&self) -> f64 {
        self.u
    }

    pub fn v(&self) -> f64 {
        self.v
    }
}

pub struct IntersecComputations<'a> {
    time: f64,
    intersected_object: &'a Object,
    world_point: Point,
    under_point: Point,
    over_point: Point,
    eye_v: Vector,
    normal_v: Vector,
    reflect_v: Vector,
    inside_obj: bool,

    refractive_from: f64,
    refractive_to: f64,
}

impl<'a> IntersecComputations<'a> {
    pub fn new(
        intersection: &'a Intersection<'a>,
        ray: &Ray,
        refractive_from: f64,
        refractive_to: f64,
    ) -> Self {
        let time = intersection.time();
        let object = intersection.object();

        let world_point = ray.position(time);
        let eye_v = -*ray.direction();
        let mut normal_v =
            object.normal_vector_at_with_intersection(world_point, Some(intersection));

        let inside_obj = normal_v.dot(eye_v) < 0.;
        if inside_obj {
            normal_v = -normal_v
        }

        let over_offset = normal_v * approx_eq::EPSILON;
        let over_point = world_point + over_offset;
        let under_point = world_point - over_offset;

        let reflect_v = ray.direction().reflect(normal_v);

        Self {
            time,
            intersected_object: object,
            world_point,
            under_point,
            over_point,
            eye_v,
            normal_v,
            reflect_v,
            inside_obj,

            refractive_from,
            refractive_to,
        }
    }

    pub fn from_intersections(
        hit: &'a Intersection<'a>,
        intersection_vec: &IntersectionCollection,
    ) -> IntersecComputations<'a> {
        let mut refractive_from = AIR_REFRACTIVE_INDEX;
        let mut refractive_to = AIR_REFRACTIVE_INDEX;

        let intersections = intersection_vec.vec();

        let mut containers: Vec<&Object> = Vec::with_capacity(intersections.len());

        for inter in intersections {
            if inter == hit && !containers.is_empty() {
                refractive_from = containers
                    .last()
                    .unwrap()
                    .material_unwrapped()
                    .refractive_index;
            }

            if let Some(idx) = containers
                .iter()
                .position(|&obj| same_obj_ref(inter.object(), obj))
            {
                containers.remove(idx);
            } else {
                containers.push(inter.object());
            }

            if inter == hit {
                if !containers.is_empty() {
                    refractive_to = containers
                        .last()
                        .unwrap()
                        .material_unwrapped()
                        .refractive_index;
                }
                break;
            }
        }
        Self::new(hit, intersection_vec.ray(), refractive_from, refractive_to)
    }

    pub fn from_intersection(intersection: &'a Intersection<'a>, ray: &Ray) -> Self {
        Self::new(
            intersection,
            ray,
            AIR_REFRACTIVE_INDEX,
            intersection.object().material_unwrapped().refractive_index,
        )
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn object(&self) -> &Object {
        self.intersected_object
    }

    pub fn material(&self) -> &Material {
        self.object().material_unwrapped()
    }

    pub fn world_point(&self) -> Point {
        self.world_point
    }

    pub fn eye_v(&self) -> Vector {
        self.eye_v
    }

    pub fn normal_v(&self) -> Vector {
        self.normal_v
    }

    pub fn inside_obj(&self) -> bool {
        self.inside_obj
    }

    pub fn over_point(&self) -> Point {
        self.over_point
    }

    pub fn reflect_v(&self) -> Vector {
        self.reflect_v
    }

    pub fn under_point(&self) -> Point {
        self.under_point
    }

    pub fn refractive_from(&self) -> f64 {
        self.refractive_from
    }

    pub fn refractive_to(&self) -> f64 {
        self.refractive_to
    }
}

pub struct IntersectionCollection<'a> {
    vec: Vec<Intersection<'a>>,
    ray: Ray,
}

impl<'a> IntersectionCollection<'a> {
    pub fn new_with_sorted_vec(ray: Ray, vec: Vec<Intersection<'a>>) -> Self {
        Self { ray, vec }
    }
    pub fn from_times_and_obj(ray: Ray, mut times: Vec<f64>, object: &'a Object) -> Self {
        times.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        Self::new_with_sorted_vec(
            ray,
            times
                .into_iter()
                .map(|time| Intersection::new(time, object))
                .collect(),
        )
    }
    pub fn from_ray_and_mult_objects(ray: Ray, objects: &'a [Object]) -> Self {
        let mut collector = IntersectionCollector::new();
        for object in objects {
            object.intersect(&ray, &mut collector);
        }

        Self::new_with_sorted_vec(ray, collector.collect_sorted())
    }
    pub fn from_ray_and_obj(ray: Ray, object: &'a Object) -> Self {
        let intersections = object.intersect_to_vec(&ray);
        Self::new_with_sorted_vec(ray, intersections)
    }

    pub fn has_intersection(&self) -> bool {
        !self.vec().is_empty()
    }

    pub fn hit(&self) -> Option<&Intersection> {
        self.vec.iter().find(|&ints| ints.time() > 0.)
    }

    fn computations(&self, intersection: &'a Intersection) -> IntersecComputations {
        IntersecComputations::from_intersections(intersection, self)
    }

    pub fn hit_computations(&self) -> Option<IntersecComputations> {
        self.hit().map(|inter| self.computations(inter))
    }

    pub fn computations_at_id(&self, id: usize) -> Option<IntersecComputations> {
        self.vec.get(id).map(|inter| self.computations(inter))
    }

    pub fn hit_pos(&self) -> Option<Point> {
        self.hit_computations().map(|comps| comps.world_point)
    }

    pub fn ray(&self) -> &Ray {
        &self.ray
    }

    pub fn count(&self) -> usize {
        self.vec.len()
    }
    pub fn vec(&self) -> &Vec<Intersection<'a>> {
        &self.vec
    }
    pub fn times_vec(&self) -> Vec<f64> {
        self.vec.iter().map(|inter| inter.time()).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts;

    use crate::approx_eq::{self, ApproxEq};
    use crate::primitive::matrix::Matrix;
    use crate::primitive::point::Point;
    use crate::primitive::tuple::Tuple;
    use crate::primitive::vector::Vector;
    use crate::render::object::shape::Shape;

    use super::super::{object::Object, ray::Ray};
    use super::*;

    #[test]
    fn intersect_sphere() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let obj = Object::primitive_with_shape(Shape::Sphere);
        let intersections = IntersectionCollection::from_ray_and_obj(ray, &obj);

        assert_eq!(intersections.count(), 2);

        let data = intersections.vec();

        assert_eq!(data[0].time(), 4.);
        assert_eq!(data[1].time(), 6.);
    }
    #[test]
    fn ray_intersects_sphere_at_tangent() {
        let ray = Ray::new(Point::new(0., 1., -5.), Vector::new(0., 0., 1.));
        let obj = Object::primitive_with_shape(Shape::Sphere);
        let intersections = IntersectionCollection::from_ray_and_obj(ray, &obj);

        assert_eq!(intersections.count(), 2);

        let data = intersections.vec();

        assert_eq!(data[0].time(), 5.);
        assert_eq!(data[0].time(), data[1].time());
    }
    #[test]
    fn ray_misses_sphere() {
        let ray = Ray::new(Point::new(0., 2., -5.), Vector::new(0., 0., 1.));
        let obj = Object::primitive_with_shape(Shape::Sphere);

        assert_eq!(
            IntersectionCollection::from_ray_and_obj(ray, &obj).count(),
            0
        );
    }
    #[test]
    fn intersect_ray_originates_inside_sphere() {
        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let obj = Object::primitive_with_shape(Shape::Sphere);
        let intersections = IntersectionCollection::from_ray_and_obj(ray, &obj);

        assert_eq!(intersections.count(), 2);

        let data = intersections.vec();

        assert_eq!(data[0].time(), -1.);
        assert_eq!(data[1].time(), 1.);
    }
    #[test]
    fn intersect_ray_behind_sphere() {
        let ray = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        let obj = Object::primitive_with_shape(Shape::Sphere);
        let intersections = IntersectionCollection::from_ray_and_obj(ray, &obj);

        assert_eq!(intersections.count(), 2);

        let data = intersections.vec();

        assert_eq!(data[0].time(), -6.);
        assert_eq!(data[1].time(), -4.);
    }

    #[test]
    fn intersection_hit_all_times_positive() {
        let sphere = Shape::Sphere;
        let obj = Object::primitive_with_shape(sphere);
        let ray = Ray::new(Point::zero(), Vector::zero());

        let intersections = IntersectionCollection::from_times_and_obj(ray, vec![1., 2.], &obj);
        let hit = intersections.hit();

        assert!(hit.is_some());
        assert_eq!(hit.unwrap().time(), 1.);
    }
    #[test]
    fn intersection_hit_with_negative_time() {
        let sphere = Shape::Sphere;
        let obj = Object::primitive_with_shape(sphere);
        let ray = Ray::new(Point::zero(), Vector::zero());

        let intersections = IntersectionCollection::from_times_and_obj(ray, vec![1., -1.], &obj);
        let hit = intersections.hit();

        assert!(hit.is_some());
        assert_eq!(hit.unwrap().time(), 1.);
    }
    #[test]
    fn intersection_hit_all_times_negative() {
        let sphere = Shape::Sphere;
        let obj = Object::primitive_with_shape(sphere);
        let ray = Ray::new(Point::zero(), Vector::zero());

        let intersections = IntersectionCollection::from_times_and_obj(ray, vec![-2., -1.], &obj);
        let hit = intersections.hit();

        assert!(hit.is_none());
    }
    #[test]
    fn intersection_hit_always_smallest_nonnegative() {
        let sphere = Shape::Sphere;
        let obj = Object::primitive_with_shape(sphere);
        let ray = Ray::new(Point::zero(), Vector::zero());

        let intersections =
            IntersectionCollection::from_times_and_obj(ray, vec![5., 7., -3., 2.], &obj);
        let hit = intersections.hit();

        assert!(hit.is_some());
        assert_eq!(hit.unwrap().time(), 2.);
    }
    #[test]
    fn intersec_comps_outside_obj() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let obj = Object::primitive_with_shape(Shape::Sphere);

        let inter_vec = IntersectionCollection::from_ray_and_obj(ray.clone(), &obj);
        let comps = inter_vec.hit().unwrap().computations(&ray);
        assert!(!comps.inside_obj());
    }
    #[test]
    fn intersec_comps_inside_obj() {
        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let obj = Object::primitive_with_shape(Shape::Sphere);

        let inter_vec = IntersectionCollection::from_ray_and_obj(ray.clone(), &obj);
        let comps = inter_vec.hit().unwrap().computations(&ray);

        assert!(comps.inside_obj());
        assert_eq!(comps.world_point(), Point::new(0., 0., 1.));
        assert_eq!(comps.eye_v(), Vector::new(0., 0., -1.));

        // normal is inverted
        assert_eq!(comps.normal_v(), Vector::new(0., 0., -1.));
    }

    #[test]
    fn hit_should_offset_point() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let obj =
            Object::primitive_with_transformation(Shape::Sphere, Matrix::translation(0., 0., 10.));

        let inter = Intersection::new(5., &obj);
        let comps = inter.computations(&ray);

        assert!(comps.over_point().z() < -approx_eq::EPSILON / 2.);
        assert!(comps.world_point().z() > comps.over_point().z())
    }

    #[test]
    fn intersect_plane_with_parallel_ray() {
        let plane = Object::primitive_with_shape(Shape::Plane);
        let ray = Ray::new(Point::new(0., 10., 0.), Vector::new(0., 0., 1.));

        let intersections = IntersectionCollection::from_ray_and_obj(ray, &plane);
        assert!(!intersections.has_intersection());
    }

    #[test]
    fn intersect_plane_with_coplanar_ray() {
        let plane = Object::primitive_with_shape(Shape::Plane);
        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));

        let intersections = IntersectionCollection::from_ray_and_obj(ray, &plane);
        assert!(!intersections.has_intersection());
    }

    #[test]
    fn precomputing_refletion_vecctor() {
        let plane = Object::primitive_with_shape(Shape::Plane);
        let half_sqrt = consts::FRAC_1_SQRT_2;
        let r = Ray::new(
            Point::new(0., 1., -1.),
            Vector::new(0., -half_sqrt, half_sqrt),
        );
        let i = Intersection::new(2. * half_sqrt, &plane);
        let comps = i.computations(&r);

        assert_eq!(comps.reflect_v(), Vector::new(0., half_sqrt, half_sqrt));
    }

    #[test]
    fn finding_reflective_exiting_entering_various_intersections() {
        let sphere_a = Object::primitive(
            Shape::Sphere,
            Material::glass(),
            Matrix::scaling(2., 2., 2.),
        );
        let sphere_b = Object::primitive(
            Shape::Sphere,
            Material {
                refractive_index: 2.,
                ..Material::glass()
            },
            Matrix::translation(0., 0., -0.25),
        );
        let sphere_c = Object::primitive(
            Shape::Sphere,
            Material {
                refractive_index: 2.5,
                ..Material::glass()
            },
            Matrix::translation(0., 0., 0.25),
        );

        let expected_reflective = [
            (1., 1.5),
            (1.5, 2.),
            (2., 2.5),
            (2.5, 2.5),
            (2.5, 1.5),
            (1.5, 1.),
        ];

        let ray = Ray::new(Point::zero(), Vector::new(0., 0., 1.));
        let intersections = [
            Intersection::new(2., &sphere_a),
            Intersection::new(2.75, &sphere_b),
            Intersection::new(3.25, &sphere_c),
            Intersection::new(4.75, &sphere_b),
            Intersection::new(5.25, &sphere_c),
            Intersection::new(6., &sphere_a),
        ];

        let intersections =
            IntersectionCollection::new_with_sorted_vec(ray, intersections.to_vec());

        for (i, (from, to)) in expected_reflective.iter().enumerate() {
            let comps =
                IntersecComputations::from_intersections(&intersections.vec()[i], &intersections);

            assert_eq!(comps.refractive_from, *from);
            assert_eq!(comps.refractive_to, *to);
        }
    }

    #[test]
    fn under_point_if_offset_below_surface() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let sphere = Object::primitive(
            Shape::Sphere,
            Material::glass(),
            Matrix::translation(0., 0., 1.),
        );

        let inter = Intersection::new(5., &sphere);
        let comps = inter.computations(&ray);

        assert!(comps.under_point().z() > approx_eq::EPSILON / 2.);
        assert!(comps.world_point().z() < comps.under_point().z());
    }

    #[test]
    fn over_under_points_dont_approx_eq_actual_points() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let sphere = Object::primitive(
            Shape::Sphere,
            Material::glass(),
            Matrix::translation(0., 0., 1.),
        );

        let inter = Intersection::new(5., &sphere);
        let comps = inter.computations(&ray);

        assert!(!comps.over_point().approx_eq(&comps.world_point()));
        assert!(!comps.under_point().approx_eq(&comps.world_point()));
    }
}
