use crate::{
    approx_eq::{self},
    primitive::{point::Point, vector::Vector},
};

use super::{
    material::{Material, AIR_REFRACTIVE_INDEX},
    object::Object,
    ray::Ray,
};

#[derive(Clone, Copy)]
pub struct Intersection<'a> {
    time: f64,
    intersected_object: &'a Object,
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
        }
    }

    pub fn time(&self) -> f64 {
        self.time
    }
    pub fn object(&self) -> &'a Object {
        self.intersected_object
    }

    pub fn computations(&self, ray: &Ray) -> IntersecComputations {
        IntersecComputations::from_intersection(*self, ray)
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
        time: f64,
        object: &'a Object,
        ray: &Ray,
        refractive_from: f64,
        refractive_to: f64,
    ) -> Self {
        let world_point = ray.position(time);
        let eye_v = -*ray.direction();
        let mut normal_v = object.normal_vector_at(world_point);

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
        hit: &Intersection<'a>,
        intersection_vec: &IntersecVec,
    ) -> IntersecComputations<'a> {
        let mut refractive_from = AIR_REFRACTIVE_INDEX;
        let mut refractive_to = AIR_REFRACTIVE_INDEX;

        let intersections = intersection_vec.data();

        let mut containers: Vec<&Object> = Vec::with_capacity(intersections.len());

        for inter in intersections {
            if inter == hit && !containers.is_empty() {
                refractive_from = containers.last().unwrap().material().refractive_index;
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
                    refractive_to = containers.last().unwrap().material().refractive_index;
                }
                break;
            }
        }
        Self::new(
            hit.time(),
            hit.object(),
            intersection_vec.ray(),
            refractive_from,
            refractive_to,
        )
    }

    pub fn from_intersection(intersection: Intersection<'a>, ray: &Ray) -> Self {
        Self::new(
            intersection.time,
            intersection.object(),
            ray,
            AIR_REFRACTIVE_INDEX,
            intersection.object().material().refractive_index,
        )
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn object(&self) -> &Object {
        self.intersected_object
    }

    pub fn material(&self) -> &Material {
        self.object().material()
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

pub struct IntersecVec<'a> {
    ray: Ray,
    vec: Vec<Intersection<'a>>,
}

impl<'a> IntersecVec<'a> {
    pub fn new(ray: Ray, mut vec: Vec<Intersection<'a>>) -> Self {
        vec.sort_unstable_by(|i1, i2| i1.time().partial_cmp(&i2.time()).unwrap());
        Self { vec, ray }
    }
    pub fn from_times_and_obj(ray: Ray, times: Vec<f64>, object: &'a Object) -> Self {
        Self::new(
            ray,
            times
                .into_iter()
                .map(|time| Intersection::new(time, object))
                .collect(),
        )
    }
    pub fn from_ray_and_mult_objects(ray: Ray, objects: &'a [Object]) -> Self {
        let intersections: Vec<Intersection> = objects
            .iter()
            .flat_map(|object| {
                object
                    .intersection_times(&ray)
                    .into_iter()
                    .map(|time| Intersection::new(time, object))
            })
            .collect();

        Self::new(ray, intersections)
    }
    pub fn from_ray_and_obj(ray: Ray, object: &'a Object) -> Self {
        let times = object.intersection_times(&ray);
        Self::from_times_and_obj(ray, times, object)
    }

    pub fn has_intersection(&self) -> bool {
        !self.data().is_empty()
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
    pub fn data(&self) -> &Vec<Intersection<'a>> {
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
    use crate::render::object::Shape;

    use super::super::{object::Object, ray::Ray};
    use super::*;

    #[test]
    fn intersect_sphere() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let obj = Object::with_shape(Shape::Sphere);
        let intersections = IntersecVec::from_ray_and_obj(ray, &obj);

        assert_eq!(intersections.count(), 2);

        let data = intersections.data();

        assert_eq!(data[0].time(), 4.);
        assert_eq!(data[1].time(), 6.);
    }
    #[test]
    fn ray_intersects_sphere_at_tangent() {
        let ray = Ray::new(Point::new(0., 1., -5.), Vector::new(0., 0., 1.));
        let obj = Object::with_shape(Shape::Sphere);
        let intersections = IntersecVec::from_ray_and_obj(ray, &obj);

        assert_eq!(intersections.count(), 2);

        let data = intersections.data();

        assert_eq!(data[0].time(), 5.);
        assert_eq!(data[0].time(), data[1].time());
    }
    #[test]
    fn ray_misses_sphere() {
        let ray = Ray::new(Point::new(0., 2., -5.), Vector::new(0., 0., 1.));
        let obj = Object::with_shape(Shape::Sphere);

        assert_eq!(IntersecVec::from_ray_and_obj(ray, &obj).count(), 0);
    }
    #[test]
    fn intersect_ray_originates_inside_sphere() {
        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let obj = Object::with_shape(Shape::Sphere);
        let intersections = IntersecVec::from_ray_and_obj(ray, &obj);

        assert_eq!(intersections.count(), 2);

        let data = intersections.data();

        assert_eq!(data[0].time(), -1.);
        assert_eq!(data[1].time(), 1.);
    }
    #[test]
    fn intersect_ray_behind_sphere() {
        let ray = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        let obj = Object::with_shape(Shape::Sphere);
        let intersections = IntersecVec::from_ray_and_obj(ray, &obj);

        assert_eq!(intersections.count(), 2);

        let data = intersections.data();

        assert_eq!(data[0].time(), -6.);
        assert_eq!(data[1].time(), -4.);
    }

    #[test]
    fn intersection_hit_all_times_positive() {
        let sphere = Shape::Sphere;
        let obj = Object::with_shape(sphere);
        let ray = Ray::new(Point::zero(), Vector::zero());

        let intersections = IntersecVec::from_times_and_obj(ray, vec![1., 2.], &obj);
        let hit = intersections.hit();

        assert!(hit.is_some());
        assert_eq!(hit.unwrap().time(), 1.);
    }
    #[test]
    fn intersection_hit_with_negative_time() {
        let sphere = Shape::Sphere;
        let obj = Object::with_shape(sphere);
        let ray = Ray::new(Point::zero(), Vector::zero());

        let intersections = IntersecVec::from_times_and_obj(ray, vec![1., -1.], &obj);
        let hit = intersections.hit();

        assert!(hit.is_some());
        assert_eq!(hit.unwrap().time(), 1.);
    }
    #[test]
    fn intersection_hit_all_times_negative() {
        let sphere = Shape::Sphere;
        let obj = Object::with_shape(sphere);
        let ray = Ray::new(Point::zero(), Vector::zero());

        let intersections = IntersecVec::from_times_and_obj(ray, vec![-2., -1.], &obj);
        let hit = intersections.hit();

        assert!(hit.is_none());
    }
    #[test]
    fn intersection_hit_always_smallest_nonnegative() {
        let sphere = Shape::Sphere;
        let obj = Object::with_shape(sphere);
        let ray = Ray::new(Point::zero(), Vector::zero());

        let intersections = IntersecVec::from_times_and_obj(ray, vec![5., 7., -3., 2.], &obj);
        let hit = intersections.hit();

        assert!(hit.is_some());
        assert_eq!(hit.unwrap().time(), 2.);
    }
    #[test]
    fn intersec_comps_outside_obj() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let obj = Object::with_shape(Shape::Sphere);

        let inter_vec = IntersecVec::from_ray_and_obj(ray.clone(), &obj);
        let comps = inter_vec.hit().unwrap().computations(&ray);
        assert!(!comps.inside_obj());
    }
    #[test]
    fn intersec_comps_inside_obj() {
        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let obj = Object::with_shape(Shape::Sphere);

        let inter_vec = IntersecVec::from_ray_and_obj(ray.clone(), &obj);
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
        let obj = Object::with_transformation(Shape::Sphere, Matrix::translation(0., 0., 10.));

        let inter = Intersection::new(5., &obj);
        let comps = inter.computations(&ray);

        assert!(comps.over_point().z() < -approx_eq::EPSILON / 2.);
        assert!(comps.world_point().z() > comps.over_point().z())
    }

    #[test]
    fn intersect_plane_with_parallel_ray() {
        let plane = Object::with_shape(Shape::Plane);
        let ray = Ray::new(Point::new(0., 10., 0.), Vector::new(0., 0., 1.));

        let intersections = IntersecVec::from_ray_and_obj(ray, &plane);
        assert!(!intersections.has_intersection());
    }

    #[test]
    fn intersect_plane_with_coplanar_ray() {
        let plane = Object::with_shape(Shape::Plane);
        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));

        let intersections = IntersecVec::from_ray_and_obj(ray, &plane);
        assert!(!intersections.has_intersection());
    }

    #[test]
    fn precomputing_refletion_vecctor() {
        let plane = Object::with_shape(Shape::Plane);
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
        let sphere_a = Object::new(
            Shape::Sphere,
            Material::glass(),
            Matrix::scaling(2., 2., 2.),
        );
        let sphere_b = Object::new(
            Shape::Sphere,
            Material {
                refractive_index: 2.,
                ..Material::glass()
            },
            Matrix::translation(0., 0., -0.25),
        );
        let sphere_c = Object::new(
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

        let intersections = IntersecVec::new(ray, intersections.to_vec());

        for (i, (from, to)) in expected_reflective.iter().enumerate() {
            let comps =
                IntersecComputations::from_intersections(&intersections.data()[i], &intersections);

            assert_eq!(comps.refractive_from, *from);
            assert_eq!(comps.refractive_to, *to);
        }
    }

    #[test]
    fn under_point_if_offset_below_surface() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let sphere = Object::new(
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
        let sphere = Object::new(
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
