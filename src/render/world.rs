use crate::{
    approx_eq::ApproxEq,
    primitive::{point::Point, tuple::Tuple},
    transformation::scaling_matrix,
};

use super::{
    camera::Camera,
    canvas::Canvas,
    color::Color,
    intersection::{IntersecComputations, IntersecVec},
    light::{color_of_illuminated_point, schlick_reflectance, PointLightSource},
    material::Material,
    object::Object,
    ray::Ray,
};
use super::{object::Shape, pattern::Pattern};

const MAX_RECURSIVE_DEPTH: usize = 5 - 1;

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

    fn color_at_depth(&self, ray: Ray, depth: usize) -> Color {
        self.intersect(ray)
            .hit_computations()
            .map_or(Color::black(), |hit_comps| self.shade_hit(hit_comps, depth))
    }

    pub fn color_at(&self, ray: Ray) -> Color {
        self.color_at_depth(ray, 0)
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

    pub fn render(&self, camera: &Camera) -> Canvas {
        let mut image = camera.canvas();

        image.set_each_pixel(|x: usize, y: usize| self.color_at(camera.ray_for_pixel(x, y)));
        image
    }

    pub fn light_sources(&self) -> &[PointLightSource] {
        self.light_sources.as_ref()
    }

    pub fn is_point_shadowed(&self, light_source: &PointLightSource, point: Point) -> bool {
        let v = light_source.position() - point;

        let distance = v.magnitude();
        let direction = v.normalize();

        let ray = Ray::new(point, direction);
        let intersections = self.intersect(ray);

        match intersections.hit() {
            None => false,
            Some(inter) => inter.time() < distance,
        }
    }

    fn reflected_color(&self, hit_comps: &IntersecComputations, depth: usize) -> Color {
        if depth >= MAX_RECURSIVE_DEPTH || hit_comps.object().material().reflectivity.approx_eq(&0.)
        {
            return Color::black();
        }
        let reflected_ray = Ray::new(hit_comps.over_point(), hit_comps.reflect_v());
        let color = self.color_at_depth(reflected_ray, depth + 1);

        color * hit_comps.object().material().reflectivity
    }

    fn refracted_color(&self, hit_comps: &IntersecComputations, depth: usize) -> Color {
        if depth >= MAX_RECURSIVE_DEPTH || hit_comps.object().material().transparency.approx_eq(&0.)
        {
            return Color::black();
        }

        let refraction_ratio = hit_comps.refractive_from() / hit_comps.refractive_to();
        let cos_i = hit_comps.eye_v().dot(hit_comps.normal_v());

        let sin2_t = refraction_ratio.powi(2) * (1. - cos_i.powi(2));

        if sin2_t > 1. {
            return Color::black();
        }

        let cos_t = (1. - sin2_t).sqrt();
        let direction = hit_comps.normal_v() * (refraction_ratio * cos_i - cos_t)
            - hit_comps.eye_v() * refraction_ratio;
        let refracted_ray = Ray::new(hit_comps.under_point(), direction);

        let color = self.color_at_depth(refracted_ray, depth + 1);
        color * hit_comps.object().material().transparency
    }

    pub fn shade_hit(&self, hit_comps: IntersecComputations, depth: usize) -> Color {
        self.light_sources()
            .iter()
            .fold(Color::black(), |acc, light_source| {
                let surface = color_of_illuminated_point(
                    hit_comps.object(),
                    light_source,
                    hit_comps.over_point(),
                    hit_comps.eye_v(),
                    hit_comps.normal_v(),
                    self.is_point_shadowed(light_source, hit_comps.over_point()),
                );
                let reflected = self.reflected_color(&hit_comps, depth);
                let refracted = self.refracted_color(&hit_comps, depth);

                let material = hit_comps.object().material();

                let use_schlick = material.reflectivity > 0.
                    && material.transparency > 0.
                    && !material.reflectivity.approx_eq(&0.)
                    && !material.transparency.approx_eq(&0.);

                let reflected_refracted = if use_schlick {
                    let reflectance = schlick_reflectance(&hit_comps);
                    reflected * reflectance + refracted * (1. - reflectance)
                } else {
                    reflected + refracted
                };
                acc + surface + reflected_refracted
            })
    }
}

impl Default for World {
    fn default() -> Self {
        let sphere1 = Object::with_shape_material(
            Shape::Sphere,
            Material {
                pattern: Pattern::Const(Color::new(0.8, 1.0, 0.6)),
                ambient: 0.1,
                diffuse: 0.7,
                specular: 0.2,
                ..Default::default()
            },
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
    use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

    use crate::{
        primitive::vector::Vector, render::intersection::Intersection,
        transformation::translation_matrix,
    };

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

    #[test]
    fn no_shadow_when_nothing_blocks_light() {
        let world = World::default();
        let point = Point::new(0., 10., 0.);

        assert!(!world.is_point_shadowed(&world.light_sources()[0], point))
    }

    #[test]
    fn shadow_when_object_is_between_point_and_light() {
        let world = World::default();
        let point = Point::new(10., -10., 10.);

        assert!(world.is_point_shadowed(&world.light_sources()[0], point))
    }

    #[test]
    fn no_shadow_when_object_is_behind_light() {
        let world = World::default();
        let point = Point::new(-20., 20., -20.);

        assert!(!world.is_point_shadowed(&world.light_sources()[0], point))
    }

    #[test]
    fn shade_hit_intersection_in_shadow() {
        let mut world = World::empty();
        world.add_light(PointLightSource::new(
            Point::new(0., 0., -10.),
            Color::white(),
        ));

        world.add_obj(Object::with_shape(Shape::Sphere));
        world.add_obj(Object::with_transformation(
            Shape::Sphere,
            translation_matrix(0., 0., 10.),
        ));

        let ray = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        let inter = Intersection::new(4., &world.objects[1]);
        let comps = inter.computations(&ray);

        assert_eq!(world.shade_hit(comps, 0), Color::new(0.1, 0.1, 0.1));
    }

    #[test]
    fn reflected_color_for_non_reflective_material() {
        let mut w = World::default();
        let r = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let shape = &mut w.objects[1];
        shape.material_mut().ambient = 1.;

        let i = Intersection::new(1., &w.objects[1]);
        let comps = i.computations(&r);

        assert_eq!(w.reflected_color(&comps, 0), Color::black());
    }

    #[test]
    fn shade_hit_with_reflective_material() {
        let mut w = World::default();
        let plane = Object::new(
            Shape::Plane,
            Material {
                reflectivity: 0.5,
                ..Default::default()
            },
            translation_matrix(0., -1., 0.),
        );
        w.add_obj(plane);

        let r = Ray::new(
            Point::new(0., 0., -3.),
            Vector::new(0., -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        );
        let i = Intersection::new(SQRT_2, w.objects.last().unwrap());
        let comps = i.computations(&r);

        assert!(w
            .shade_hit(comps, 0)
            .approx_eq_low_prec(&Color::new(0.87677, 0.92436, 0.82918)));
    }

    #[test]
    fn ray_bouncing_between_mutually_reflective_surfaces() {
        let mut w = World::empty();

        w.add_light(PointLightSource::new(
            Point::new(0., 0., 0.),
            Color::white(),
        ));

        let lower = Object::new(
            Shape::Plane,
            Material {
                reflectivity: 1.,
                ..Default::default()
            },
            translation_matrix(0., -1., 0.),
        );
        let upper = Object::new(
            Shape::Plane,
            Material {
                reflectivity: 1.,
                ..Default::default()
            },
            translation_matrix(0., 1., 0.),
        );
        w.add_obj(lower);
        w.add_obj(upper);

        let r = Ray::new(Point::zero(), Vector::new(0., 1., 0.));

        let _ = w.color_at(r);
    }

    #[test]
    fn reflected_color_at_max_recursive_depth() {
        let mut w = World::default();
        let plane = Object::new(
            Shape::Plane,
            Material {
                reflectivity: 0.5,
                ..Default::default()
            },
            translation_matrix(0., -1., 0.),
        );
        w.add_obj(plane);

        let r = Ray::new(
            Point::new(0., 0., -3.),
            Vector::new(0., -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        );
        let i = Intersection::new(SQRT_2, w.objects.last().unwrap());
        let comps = i.computations(&r);

        assert_eq!(
            w.reflected_color(&comps, MAX_RECURSIVE_DEPTH),
            Color::black()
        );
    }

    #[test]
    fn refraced_colr_with_opaque_surface() {
        let world = World::default();
        let shape = &world.objects[0];
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let intersections = IntersecVec::from_times_and_obj(ray, vec![4., 6.], shape);
        let comps = intersections.hit_computations().unwrap();

        assert_eq!(world.refracted_color(&comps, 0), Color::black());
    }

    #[test]
    fn refracted_color_at_max_recursive_depth() {
        let mut world = World::default();
        let shape = &mut world.objects[0];
        shape.material_mut().transparency = 1.;
        shape.material_mut().refractive_index = 1.5;
        let shape = &world.objects[0];

        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let intersections = IntersecVec::from_times_and_obj(ray, vec![4., 6.], shape);
        let comps = intersections.hit_computations().unwrap();

        assert_eq!(
            world.refracted_color(&comps, MAX_RECURSIVE_DEPTH),
            Color::black()
        );
    }

    #[test]
    fn refracted_color_under_total_internal_reflection() {
        let mut world = World::default();
        let shape = &mut world.objects[0];
        shape.material_mut().transparency = 1.;
        shape.material_mut().refractive_index = 1.5;
        let shape = &world.objects[0];

        let ray = Ray::new(Point::new(0., 0., SQRT_2 / 2.), Vector::new(0., 1., 0.));
        let intersections =
            IntersecVec::from_times_and_obj(ray, vec![-SQRT_2 / 2., SQRT_2 / 2.], shape);

        let comps = intersections.computations_at_id(1).unwrap();

        assert_eq!(world.refracted_color(&comps, 0), Color::black());
    }

    #[test]
    fn refracted_color_with_refracted_ray() {
        let mut world = World::default();

        let a = &mut world.objects[0];
        a.material_mut().ambient = 1.;
        a.material_mut().pattern = Pattern::test_pattern(None);

        let b = &mut world.objects[1];
        b.material_mut().transparency = 1.;
        b.material_mut().refractive_index = 1.5;

        let ray = Ray::new(Point::new(0., 0., 0.1), Vector::new(0., 1., 0.));

        let a = &world.objects[0];
        let b = &world.objects[1];
        let objects = vec![a.clone(), b.clone()];

        let intersections = IntersecVec::from_ray_and_mult_objects(ray, &objects);
        let comps = intersections.hit_computations().unwrap();

        assert!(world
            .refracted_color(&comps, 0)
            .approx_eq_low_prec(&Color::new(0., 0.99888, 0.04725)));
    }

    #[test]
    fn shading_transparent_material() {
        let mut world = World::default();
        let floor = Object::new(
            Shape::Plane,
            Material {
                transparency: 0.5,
                refractive_index: 1.5,
                ..Default::default()
            },
            translation_matrix(0., -1., 0.),
        );
        let ball = Object::new(
            Shape::Sphere,
            Material {
                pattern: Pattern::Const(Color::red()),
                ambient: 0.5,
                ..Default::default()
            },
            translation_matrix(0., -3.5, -0.5),
        );
        world.add_obj(floor);
        world.add_obj(ball);

        let ray = Ray::new(
            Point::new(0., 0., -3.),
            Vector::new(0., -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        );
        let intersections = world.intersect(ray);
        let cmps = intersections.hit_computations().unwrap();

        assert_eq!(
            world.shade_hit(cmps, 0),
            Color::new(0.93642, 0.68642, 0.68642)
        );
    }

    #[test]
    fn shading_reflective_transparent_material() {
        let mut world = World::default();
        let floor = Object::new(
            Shape::Plane,
            Material {
                transparency: 0.5,
                reflectivity: 0.5,
                refractive_index: 1.5,
                ..Default::default()
            },
            translation_matrix(0., -1., 0.),
        );
        let ball = Object::new(
            Shape::Sphere,
            Material {
                pattern: Pattern::Const(Color::red()),
                ambient: 0.5,
                ..Default::default()
            },
            translation_matrix(0., -3.5, -0.5),
        );
        world.add_obj(floor);
        world.add_obj(ball);

        let ray = Ray::new(
            Point::new(0., 0., -3.),
            Vector::new(0., -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        );
        let intersections = world.intersect(ray);
        let cmps = intersections.hit_computations().unwrap();

        assert_eq!(
            world.shade_hit(cmps, 0),
            Color::new(0.93391, 0.69643, 0.69243)
        );
    }
}
