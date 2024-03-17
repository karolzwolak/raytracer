use crate::{
    approx_eq::ApproxEq,
    primitive::{matrix::Matrix, point::Point, tuple::Tuple},
    render::object::group::ObjectGroup,
};

use super::{
    camera::Camera,
    canvas::Canvas,
    color::Color,
    light::{color_of_illuminated_point, schlick_reflectance, PointLightSource},
    material::Material,
    object::Object,
    ray::Ray,
};
use super::{
    intersection::{IntersecComputations, IntersectionCollection},
    object::PrimitiveObject,
};
use super::{object::shape::Shape, pattern::Pattern};

pub struct World {
    objects: ObjectGroup,
    light_sources: Vec<PointLightSource>,
    /// Depth of recursive calls for reflections and refractions
    /// 0 means no reflections or refractions
    max_recursive_depth: usize,
    /// offset from the center of the pixel
    /// so it should be in range [-0.5, 0.5]
    // TODO: intelligently apply supersampling only if it makes a difference
    supersampling_offsets: Vec<f64>,
    /// If true, shadows are calculated with intensity,
    /// so that all objects don't cast full shadow
    /// boolean shadows are required for testing purposes,
    /// because all tests values were calculated with bool shadows
    use_shadow_intensity: bool,
}

impl World {
    const MAX_RECURSIVE_DEPTH: usize = 5 - 1;
    const DEFAULT_SUPERSAMPLING_LEVEL: usize = 2;
    pub const MAX_DIM: f64 = 10.0e6;

    fn gen_supersampling_offsets(level: usize) -> Vec<f64> {
        match level {
            0 | 1 => vec![0.],
            2 => vec![-0.25, 0.25],
            3 => vec![-0.25, 0., 0.25],
            _ => vec![-0.5, -0.25, -0.125, 0., 0.125, 0.25, 0.5],
        }
    }

    pub fn with_supersampling_level(
        objects: Vec<Object>,
        light_sources: Vec<PointLightSource>,
        supersampling_level: Option<usize>,
        max_recursive_depth: Option<usize>,
        use_shadow_intensity: bool,
    ) -> Self {
        let mut objects = ObjectGroup::new(objects);

        let now = std::time::Instant::now();
        objects.partition();
        println!("partitioning time: {:?}", now.elapsed());

        Self {
            objects,
            light_sources,
            supersampling_offsets: Self::gen_supersampling_offsets(
                supersampling_level.unwrap_or(Self::DEFAULT_SUPERSAMPLING_LEVEL),
            ),
            max_recursive_depth: max_recursive_depth.unwrap_or(Self::MAX_RECURSIVE_DEPTH),
            use_shadow_intensity,
        }
    }
    pub fn new(
        objects: Vec<Object>,
        light_sources: Vec<PointLightSource>,
        max_recursive_depth: Option<usize>,
    ) -> Self {
        Self::with_supersampling_level(objects, light_sources, None, max_recursive_depth, true)
    }

    pub fn with_bool_shadows(
        objects: Vec<Object>,
        light_sources: Vec<PointLightSource>,
        max_recursive_depth: Option<usize>,
    ) -> Self {
        Self::with_supersampling_level(objects, light_sources, Some(1), max_recursive_depth, false)
    }

    pub fn testing(objects: Vec<Object>, light_sources: Vec<PointLightSource>) -> Self {
        Self {
            objects: ObjectGroup::new(objects),
            light_sources,
            supersampling_offsets: Self::gen_supersampling_offsets(0),
            max_recursive_depth: Self::MAX_RECURSIVE_DEPTH,
            use_shadow_intensity: false,
        }
    }
    pub fn empty() -> Self {
        Self::new(Vec::new(), Vec::new(), None)
    }

    pub fn objects(&self) -> &[Object] {
        self.objects.children()
    }

    pub fn objects_mut(&mut self) -> &mut [Object] {
        self.objects.children_mut()
    }

    pub fn intersect(&self, ray: Ray) -> IntersectionCollection {
        IntersectionCollection::from_group(ray, &self.objects)
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
        self.objects.add_child(obj);
    }

    pub fn add_light(&mut self, light_source: PointLightSource) {
        self.light_sources.push(light_source);
    }

    pub fn set_light_sources(&mut self, light_sources: Vec<PointLightSource>) {
        self.light_sources = light_sources;
    }

    fn color_at_pixel(&self, x: usize, y: usize, camera: &Camera) -> Color {
        let x = x as f64;
        let y = y as f64;

        let offsets = &self.supersampling_offsets;
        let mut color = Color::black();

        for dx in offsets {
            for dy in offsets {
                color = color + self.color_at(camera.ray_for_pixel(x + dx, y + dy));
            }
        }
        color / offsets.len().pow(2) as f64
    }

    pub fn render(&self, camera: &Camera) -> Canvas {
        let mut image = camera.canvas();

        let primitive_count = self.objects.primitive_count();

        let ray_count = image.width() * image.height() * self.supersampling_offsets.len().pow(2);

        println!("rendering {} objects", primitive_count);
        println!("with {} rays", ray_count);

        let now = std::time::Instant::now();

        image.set_each_pixel(|x: usize, y: usize| self.color_at_pixel(x, y, camera));
        println!("render time: {:?}", now.elapsed());
        let rays_per_sec = ray_count as f64 / now.elapsed().as_secs_f64();
        println!("rays per second: {}", rays_per_sec.round());
        image
    }

    pub fn light_sources(&self) -> &[PointLightSource] {
        self.light_sources.as_ref()
    }

    /// 0. means no shadow, 1. means full shadow
    pub fn point_shadow_intensity(&self, light_source: &PointLightSource, point: Point) -> f64 {
        let v = light_source.position() - point;

        let distance = v.magnitude();
        let direction = v.normalize();

        let ray = Ray::new(point, direction);
        let intersections = self.intersect(ray);

        if !self.use_shadow_intensity {
            return match intersections.hit() {
                Some(inter) => {
                    if inter.time() < distance && !inter.time().approx_eq(&distance) {
                        1.0
                    } else {
                        0.
                    }
                }
                None => 0.,
            };
        }

        // calculate shadow intensity by summation of transparency of all objects
        // (1 - transparency to be exact)
        let mut intensity = 0.;
        for inter in intersections.vec() {
            // skip intersections behind light source
            if inter.time() < 0. {
                continue;
            }
            if inter.time().approx_eq(&distance) || inter.time() > distance {
                break;
            }
            intensity += 1. - inter.object().material_unwrapped().transparency;
            if intensity >= 1. {
                return 1.;
            }
        }
        intensity
    }

    fn reflected_color(&self, hit_comps: &IntersecComputations, depth: usize) -> Color {
        if depth >= self.max_recursive_depth
            || hit_comps
                .object()
                .material_unwrapped()
                .reflectivity
                .approx_eq(&0.)
        {
            return Color::black();
        }
        let reflected_ray = Ray::new(hit_comps.over_point(), hit_comps.reflect_v());
        let color = self.color_at_depth(reflected_ray, depth + 1);

        color * hit_comps.object().material_unwrapped().reflectivity
    }

    fn refracted_color(&self, hit_comps: &IntersecComputations, depth: usize) -> Color {
        if depth >= self.max_recursive_depth
            || hit_comps
                .object()
                .material_unwrapped()
                .transparency
                .approx_eq(&0.)
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
        color * hit_comps.object().material_unwrapped().transparency
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
                    self.point_shadow_intensity(light_source, hit_comps.over_point()),
                );
                let reflected = self.reflected_color(&hit_comps, depth);
                let refracted = self.refracted_color(&hit_comps, depth);

                let material = hit_comps.object().material_unwrapped();

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

// Default testing world with bool shadows
impl World {
    pub fn default_testing() -> Self {
        let sphere1 = Object::primitive(
            Shape::Sphere,
            Material {
                pattern: Pattern::Const(Color::new(0.8, 1.0, 0.6)),
                ambient: 0.1,
                diffuse: 0.7,
                specular: 0.2,
                ..Default::default()
            },
            Matrix::identity(),
        );
        let sphere2 =
            PrimitiveObject::with_transformation(Shape::Sphere, Matrix::scaling(0.5, 0.5, 0.5))
                .into();

        let objects = vec![sphere1, sphere2];
        let lights = vec![PointLightSource::new(
            Point::new(-10., 10., -10.),
            Color::white(),
        )];
        Self::testing(objects, lights)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

    use crate::{primitive::vector::Vector, render::intersection::Intersection};

    use super::*;

    #[test]
    fn intersect_world_with_ray() {
        let world = World::default_testing();
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));

        let intersections = world.intersect(ray);
        assert_eq!(intersections.times_vec(), vec![4., 4.5, 5.5, 6.]);
    }
    #[test]
    fn shade_intersection() {
        let world = World::default_testing();
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));

        assert_eq!(world.color_at(ray), Color::new(0.38066, 0.47583, 0.2855));
    }

    #[test]
    fn shade_intersection_from_inside() {
        let mut world = World::default_testing();
        world.set_light_sources(vec![PointLightSource::new(
            Point::new(0., 0.25, 0.),
            Color::new(1., 1., 1.),
        )]);

        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));

        assert_eq!(world.color_at(ray), Color::new(0.90498, 0.90498, 0.90498));
    }

    #[test]
    fn color_when_ray_misses() {
        let world = World::default_testing();
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 1., 0.));

        assert_eq!(world.color_at(ray), Color::black());
    }

    #[test]
    fn color_when_ray_hits() {
        let world = World::default_testing();
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 1., 0.));

        assert_eq!(world.color_at(ray), Color::black());
    }

    #[test]
    fn no_shadow_when_nothing_blocks_light() {
        let world = World::default_testing();
        let point = Point::new(0., 10., 0.);

        assert_eq!(
            world.point_shadow_intensity(&world.light_sources()[0], point),
            0.
        )
    }

    #[test]
    fn shadow_when_object_is_between_point_and_light() {
        let world = World::default_testing();
        let point = Point::new(10., -10., 10.);

        assert_eq!(
            world.point_shadow_intensity(&world.light_sources()[0], point),
            1.
        )
    }

    #[test]
    fn no_shadow_when_object_is_behind_light() {
        let world = World::default_testing();
        let point = Point::new(-20., 20., -20.);

        assert_eq!(
            world.point_shadow_intensity(&world.light_sources()[0], point),
            0.
        )
    }

    #[test]
    fn shade_hit_intersection_in_shadow() {
        let mut world = World::empty();
        world.add_light(PointLightSource::new(
            Point::new(0., 0., -10.),
            Color::white(),
        ));

        world.add_obj(Object::primitive_with_shape(Shape::Sphere));
        world.add_obj(Object::primitive_with_transformation(
            Shape::Sphere,
            Matrix::translation(0., 0., 10.),
        ));

        let ray = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        let inter = Intersection::new(4., &world.objects()[1]);
        let comps = inter.computations(&ray);

        assert_eq!(world.shade_hit(comps, 0), Color::new(0.1, 0.1, 0.1));
    }

    #[test]
    fn reflected_color_for_non_reflective_material() {
        let mut w = World::default_testing();
        let r = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let shape = &mut w.objects_mut()[1];
        shape.material_mut().unwrap().ambient = 1.;

        let i = Intersection::new(1., &w.objects()[1]);
        let comps = i.computations(&r);

        assert_eq!(w.reflected_color(&comps, 0), Color::black());
    }

    #[test]
    fn shade_hit_with_reflective_material() {
        let mut w = World::default_testing();
        let plane = Object::primitive(
            Shape::Plane,
            Material {
                reflectivity: 0.5,
                ..Default::default()
            },
            Matrix::translation(0., -1., 0.),
        );
        w.add_obj(plane);

        let r = Ray::new(
            Point::new(0., 0., -3.),
            Vector::new(0., -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        );
        let i = Intersection::new(SQRT_2, w.objects().last().unwrap());
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

        let lower = Object::primitive(
            Shape::Plane,
            Material {
                reflectivity: 1.,
                ..Default::default()
            },
            Matrix::translation(0., -1., 0.),
        );
        let upper = Object::primitive(
            Shape::Plane,
            Material {
                reflectivity: 1.,
                ..Default::default()
            },
            Matrix::translation(0., 1., 0.),
        );
        w.add_obj(lower);
        w.add_obj(upper);

        let r = Ray::new(Point::zero(), Vector::new(0., 1., 0.));

        let _ = w.color_at(r);
    }

    #[test]
    fn reflected_color_at_max_recursive_depth() {
        let mut world = World::default_testing();
        let plane = Object::primitive(
            Shape::Plane,
            Material {
                reflectivity: 0.5,
                ..Default::default()
            },
            Matrix::translation(0., -1., 0.),
        );
        world.add_obj(plane);

        let r = Ray::new(
            Point::new(0., 0., -3.),
            Vector::new(0., -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        );
        let i = Intersection::new(SQRT_2, world.objects().last().unwrap());
        let comps = i.computations(&r);

        assert_eq!(
            world.reflected_color(&comps, world.max_recursive_depth),
            Color::black()
        );
    }

    #[test]
    fn refraced_colr_with_opaque_surface() {
        let world = World::default_testing();
        let shape = &world.objects()[0];
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let intersections = IntersectionCollection::from_times_and_obj(ray, vec![4., 6.], shape);
        let comps = intersections.hit_computations().unwrap();

        assert_eq!(world.refracted_color(&comps, 0), Color::black());
    }

    #[test]
    fn refracted_color_at_max_recursive_depth() {
        let mut world = World::default_testing();
        let shape = &mut world.objects_mut()[0];
        shape.material_mut().unwrap().transparency = 1.;
        shape.material_mut().unwrap().refractive_index = 1.5;
        let shape = &world.objects()[0];

        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let intersections = IntersectionCollection::from_times_and_obj(ray, vec![4., 6.], shape);
        let comps = intersections.hit_computations().unwrap();

        assert_eq!(
            world.refracted_color(&comps, world.max_recursive_depth),
            Color::black()
        );
    }

    #[test]
    fn refracted_color_under_total_internal_reflection() {
        let mut world = World::default_testing();
        let shape = &mut world.objects_mut()[0];
        shape.material_mut().unwrap().transparency = 1.;
        shape.material_mut().unwrap().refractive_index = 1.5;
        let shape = &world.objects()[0];

        let ray = Ray::new(Point::new(0., 0., SQRT_2 / 2.), Vector::new(0., 1., 0.));
        let intersections =
            IntersectionCollection::from_times_and_obj(ray, vec![-SQRT_2 / 2., SQRT_2 / 2.], shape);

        let comps = intersections.computations_at_id(1).unwrap();

        assert_eq!(world.refracted_color(&comps, 0), Color::black());
    }

    #[test]
    fn refracted_color_with_refracted_ray() {
        let mut world = World::default_testing();

        let a = &mut world.objects_mut()[0];
        a.material_mut().unwrap().ambient = 1.;
        a.material_mut().unwrap().pattern = Pattern::test_pattern(None);

        let b = &mut world.objects_mut()[1];
        b.material_mut().unwrap().transparency = 1.;
        b.material_mut().unwrap().refractive_index = 1.5;

        let ray = Ray::new(Point::new(0., 0., 0.1), Vector::new(0., 1., 0.));

        let a = &world.objects()[0];
        let b = &world.objects()[1];
        let objects = vec![a.clone(), b.clone()];

        let intersections = IntersectionCollection::from_ray_and_mult_objects(ray, &objects);
        let comps = intersections.hit_computations().unwrap();

        assert!(world
            .refracted_color(&comps, 0)
            .approx_eq_low_prec(&Color::new(0., 0.99888, 0.04725)));
    }

    #[test]
    fn shading_transparent_material() {
        let mut world = World::default_testing();
        let floor = Object::primitive(
            Shape::Plane,
            Material {
                transparency: 0.5,
                refractive_index: 1.5,
                ..Default::default()
            },
            Matrix::translation(0., -1., 0.),
        );
        let ball = Object::primitive(
            Shape::Sphere,
            Material {
                pattern: Pattern::Const(Color::red()),
                ambient: 0.5,
                ..Default::default()
            },
            Matrix::translation(0., -3.5, -0.5),
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
        let mut world = World::default_testing();
        let floor = Object::primitive(
            Shape::Plane,
            Material {
                transparency: 0.5,
                reflectivity: 0.5,
                refractive_index: 1.5,
                ..Default::default()
            },
            Matrix::translation(0., -1., 0.),
        );
        let ball = Object::primitive(
            Shape::Sphere,
            Material {
                pattern: Pattern::Const(Color::red()),
                ambient: 0.5,
                ..Default::default()
            },
            Matrix::translation(0., -3.5, -0.5),
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
