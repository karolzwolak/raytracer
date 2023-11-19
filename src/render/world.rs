use crate::{
    primitive::{point::Point, tuple::Tuple},
    transformation::scaling_matrix,
};

use super::{
    color::Color, intersection::IntersecVec, light::PointLightSource, material::Material,
    object::Object, ray::Ray, shape::Shape,
};

pub struct World {
    objects: Vec<Object>,
    light_sources: Vec<PointLightSource>,
}
impl World {
    pub fn new(objects: Vec<Object>, light_sources: Vec<PointLightSource>) -> Self {
        Self {
            objects,
            light_sources,
        }
    }
    pub fn empty() -> Self {
        Self::new(Vec::new(), Vec::new())
    }
    pub fn intersect(&self, ray: Ray) -> IntersecVec {
        IntersecVec::from_ray_and_mult_objects(ray, &self.objects)
    }
    pub fn color_at(&self, ray: Ray) -> Color {
        self.intersect(ray)
            .hit_computations()
            .map_or(Color::black(), |comps| comps.shade_hit(&self.light_sources))
    }

    pub fn add_obj(&mut self, obj: Object) {
        self.objects.push(obj);
    }

    pub fn add_light(&mut self, light_source: PointLightSource) {
        self.light_sources.push(light_source);
    }

    pub fn set_objects(&mut self, objects: Vec<Object>) {
        self.objects = objects;
    }

    pub fn set_light_sources(&mut self, light_sources: Vec<PointLightSource>) {
        self.light_sources = light_sources;
    }
}

impl Default for World {
    fn default() -> Self {
        let sphere1 = Object::with_shape_material(
            Shape::Sphere,
            Material::new(Color::new(0.8, 1.0, 0.6), 0.1, 0.7, 0.2, 200.),
        );
        let sphere2 = Object::with_transformation(Shape::Sphere, scaling_matrix(0.5, 0.5, 0.5));

        let objects = vec![sphere1, sphere2];
        let lights = vec![PointLightSource::new(
            Point::new(-10., 10., -10.),
            Color::white(),
        )];
        Self::new(objects, lights)
    }
}

#[cfg(test)]
mod tests {
    use crate::primitive::vector::Vector;

    use super::*;

    #[test]
    fn intersect_world_with_ray() {
        let world = World::default();
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));

        let intersections = world.intersect(ray);
        assert_eq!(intersections.times_vec(), vec![4., 4.5, 5.5, 6.]);
    }
    #[test]
    fn shade_intersection() {
        let world = World::default();
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));

        assert_eq!(world.color_at(ray), Color::new(0.38066, 0.47583, 0.2855));
    }

    #[test]
    fn shade_intersection_from_inside() {
        let mut world = World::default();
        world.set_light_sources(vec![PointLightSource::new(
            Point::new(0., 0.25, 0.),
            Color::new(1., 1., 1.),
        )]);

        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));

        assert_eq!(world.color_at(ray), Color::new(0.90498, 0.90498, 0.90498));
    }

    #[test]
    fn color_when_ray_misses() {
        let world = World::default();
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 1., 0.));

        assert_eq!(world.color_at(ray), Color::black());
    }

    #[test]
    fn color_when_ray_hits() {
        let world = World::default();
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 1., 0.));

        assert_eq!(world.color_at(ray), Color::black());
    }
}
