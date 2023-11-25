use crate::{
    approx_eq,
    primitive::{point::Point, tuple::Tuple, vector::Vector},
    render::shape::Shape,
};

use super::{
    material::Material,
    object::Object,
    ray::Ray,
};

#[derive(Clone, Copy)]
pub struct Intersection<'a> {
    time: f64,
    intersected_object: &'a Object,
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
    over_point: Point,
    eye_v: Vector,
    normal_v: Vector,
    inside_obj: bool,
}

impl<'a> IntersecComputations<'a> {
    pub fn new(time: f64, object: &'a Object, ray: &Ray) -> Self {
        let world_point = ray.position(time);
        let eye_v = -*ray.direction();
        let mut normal_v = object.normal_vector_at(world_point);

        let inside_obj = normal_v.dot(eye_v) < 0.;
        if inside_obj {
            normal_v = -normal_v
        }

        let over_point = world_point + normal_v * approx_eq::EPSILON;

        Self {
            time,
            intersected_object: object,
            world_point,
            over_point,
            eye_v,
            normal_v,
            inside_obj,
        }
    }
    pub fn from_intersection(intersection: Intersection<'a>, ray: &Ray) -> Self {
        Self::new(intersection.time(), intersection.object(), ray)
    }

    pub fn try_from_ray_and_obj(ray: Ray, obj: &'a Object) -> Option<Self> {
        let intersections = IntersecVec::from_ray_and_obj(ray, obj);
        intersections
            .hit()
            .map(|inter| Self::new(inter.time(), obj, intersections.ray()))
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
}

// impl<'a> PartialEq for Intersection<'a> {
//     fn eq(&self, other: &Self) -> bool {
//         self.time.approq_eq(other.time) && self.intersected_object == other.intersected_object
//     }
// }

pub struct IntersecVec<'a> {
    ray: Ray,
    vec: Vec<Intersection<'a>>,
}

impl<'a> IntersecVec<'a> {
    fn intersection_times(ray: &Ray, object: &'a Object) -> Vec<f64> {
        let ray = ray.transform(object.transformation_inverse().unwrap());
        match object.shape() {
            Shape::Sphere => {
                let vector_sphere_to_ray = *ray.origin() - Point::new(0., 0., 0.);

                let a = ray.direction().dot(*ray.direction());
                let b = 2. * ray.direction().dot(vector_sphere_to_ray);
                let c = vector_sphere_to_ray.dot(vector_sphere_to_ray) - 1.;

                let discriminant = b * b - 4. * a * c;
                if discriminant < 0. || a == 0. {
                    return Vec::new();
                }

                let delta_sqrt = discriminant.sqrt();
                vec![(-b - delta_sqrt) / (2. * a), (-b + delta_sqrt) / (2. * a)]
            }
        }
    }
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
                Self::intersection_times(&ray, object)
                    .into_iter()
                    .map(|time| Intersection::new(time, object))
            })
            .collect();

        Self::new(ray, intersections)
    }
    pub fn from_ray_and_obj(ray: Ray, object: &'a Object) -> Self {
        let times = Self::intersection_times(&ray, object);
        Self::from_times_and_obj(ray, times, object)
    }

    pub fn does_intersect(ray: &Ray, object: &'a Object) -> bool {
        !Self::intersection_times(ray, object).is_empty()
    }

    pub fn has_intersection(&self) -> bool {
        !self.data().is_empty()
    }

    pub fn hit(&self) -> Option<&Intersection> {
        self.vec.iter().find(|&ints| ints.time() > 0.)
    }

    pub fn hit_computations(&self) -> Option<IntersecComputations> {
        self.hit().map(|inter| inter.computations(&self.ray))
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
    use crate::approx_eq;
    use crate::primitive::point::Point;
    use crate::primitive::tuple::Tuple;
    use crate::primitive::vector::Vector;
    use crate::render::shape::Shape;
    use crate::transformation::{scaling_matrix, translation_matrix};

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
    fn intersect_scaled_sphere() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let obj = Object::with_transformation(Shape::Sphere, scaling_matrix(2., 2., 2.));

        let int_times = IntersecVec::intersection_times(&ray, &obj);
        assert_eq!(int_times, vec![3., 7.]);
    }
    #[test]
    fn intersect_translated_sphere() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let obj = Object::with_transformation(Shape::Sphere, translation_matrix(5., 0., 0.));

        let int_times = IntersecVec::intersection_times(&ray, &obj);
        assert_eq!(int_times, vec![]);
    }

    #[test]
    fn intersec_comps_outside_obj() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let obj = Object::with_shape(Shape::Sphere);

        let comps = IntersecComputations::try_from_ray_and_obj(ray, &obj);
        assert!(comps.is_some());
        assert!(!comps.unwrap().inside_obj());
    }
    #[test]
    fn intersec_comps_inside_obj() {
        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let obj = Object::with_shape(Shape::Sphere);

        let comps = IntersecComputations::try_from_ray_and_obj(ray, &obj);
        assert!(comps.is_some());
        let comps = comps.unwrap();

        assert!(comps.inside_obj());
        assert_eq!(comps.world_point(), Point::new(0., 0., 1.));
        assert_eq!(comps.eye_v(), Vector::new(0., 0., -1.));

        // normal is inverted
        assert_eq!(comps.normal_v(), Vector::new(0., 0., -1.));
    }

    #[test]
    fn hit_should_offset_point() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let obj = Object::with_transformation(Shape::Sphere, translation_matrix(0., 0., 10.));

        let inter = Intersection::new(5., &obj);
        let comps = inter.computations(&ray);

        assert!(comps.over_point().z() < -approx_eq::EPSILON / 2.);
        assert!(comps.world_point().z() > comps.over_point().z())
    }
}
