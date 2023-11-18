use crate::{
    primitive::{point::Point, tuple::Tuple},
    render::shape::Shape,
};

use super::{object::Object, ray::Ray};

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

    pub fn intersection_point(&self, intersection: &Intersection) -> Point {
        self.ray.position(intersection.time())
    }
    pub fn hit(&self) -> Option<&Intersection> {
        self.vec.iter().find(|&ints| ints.time() > 0.)
    }

    pub fn hit_pos(&self) -> Option<Point> {
        self.hit().map(|inter| self.intersection_point(inter))
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
}

#[cfg(test)]
mod tests {
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
}
