pub mod animation;
pub mod camera;
pub mod io;
pub mod light;
pub mod object;

use crate::scene::camera::Camera;
use derive_builder::Builder;
use light::schlick_reflectance;
use object::{group::ObjectGroup, PrimitiveObject};

use crate::{
    approx_eq::ApproxEq,
    math::{matrix::Matrix, point::Point, tuple::Tuple, Color},
    render::{canvas::Canvas, intersection::IntersectionCollector, ray::Ray},
    Material, Object, Pattern, PointLightSource, Shape,
};

use crate::render::intersection::{IntersecComputations, IntersectionCollection};

#[derive(PartialEq, Debug, Clone, Builder)]
#[builder(default)]
pub struct Scene {
    objects: ObjectGroup,
    light_sources: Vec<PointLightSource>,
    /// Depth of recursive calls for reflections and refractions
    /// 0 means no reflections or refractions
    #[builder(default = "Scene::MAX_RECURSIVE_DEPTH")]
    max_recursive_depth: usize,
    /// offset from the center of the pixel
    /// so it should be in range [-0.5, 0.5]
    // TODO: intelligently apply supersampling only if it makes a difference
    #[builder(setter(custom))]
    #[builder(field(
        ty = "Option<usize>",
        build = "Scene::gen_supersampling_offsets(self.supersampling_offsets.unwrap_or(Scene::DEFAULT_SUPERSAMPLING_LEVEL))"
    ))]
    supersampling_offsets: Vec<f64>,
    /// If true, shadows are calculated with intensity,
    /// so that all objects don't cast full shadow
    /// boolean shadows are required for testing purposes,
    /// because all tests values were calculated with bool shadows
    #[builder(default = "true")]
    use_shadow_intensity: bool,
}

impl Default for Scene {
    fn default() -> Self {
        Self::empty()
    }
}

impl SceneBuilder {
    fn supersampling_level(&mut self, level: usize) -> &mut Self {
        self.supersampling_offsets = Some(level);
        self
    }
}

impl Scene {
    const MAX_RECURSIVE_DEPTH: usize = 5 - 1;
    const DEFAULT_SUPERSAMPLING_LEVEL: usize = 2;
    pub const MAX_DIM: f64 = 10.0e6;

    fn gen_supersampling_offsets(level: usize) -> Vec<f64> {
        match level {
            0 | 1 => vec![0.],
            2 => vec![-0.25, 0.25],
            3 => vec![-0.25, 0., 0.25],
            4 => vec![-0.5, -0.25, 0.25, 0.5],
            _ => vec![-0.5, -0.25, 0., 0.25, 0.5],
        }
    }

    pub fn with_supersampling_level(
        objects: Vec<Object>,
        light_sources: Vec<PointLightSource>,
        supersampling_level: Option<usize>,
        max_recursive_depth: Option<usize>,
        use_shadow_intensity: bool,
    ) -> Self {
        let objects = ObjectGroup::new(objects);

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

    pub fn intersect_testing(&self, ray: Ray) -> IntersectionCollection {
        let mut collector = IntersectionCollector::new_keep_redundant();
        self.objects.intersect(&ray, &mut collector);
        IntersectionCollection::from_collector(ray, collector)
    }

    pub fn intersect_with_dest_obj<'a>(
        &'a self,
        ray: Ray,
        obj: &'a Object,
    ) -> IntersectionCollection<'a> {
        let mut collector = IntersectionCollector::with_dest_obj(&ray, obj);
        self.objects.intersect(&ray, &mut collector);
        IntersectionCollection::from_collector(ray, collector)
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

    pub fn render_animation_frame(
        &mut self,
        camera: &Camera,
        progressbar: indicatif::ProgressBar,
    ) -> Canvas {
        self.objects.build_bvh();

        let mut image = camera.canvas();

        image.set_each_pixel(
            |x: usize, y: usize| self.color_at_pixel(x, y, camera),
            progressbar,
        );
        image
    }

    pub fn render(&mut self, camera: &Camera) -> Canvas {
        let now = std::time::Instant::now();
        self.objects.build_bvh();
        println!("partitioning time: {:?}", now.elapsed());

        let mut image = camera.canvas();

        let primitive_count = self.objects.primitive_count();

        let ray_count = image.width() * image.height() * self.supersampling_offsets.len().pow(2);

        println!(
            "rendering image with {}x{} resolution",
            image.width(),
            image.height()
        );
        println!("rendering {} objects", primitive_count);
        println!("with {} rays", ray_count);
        println!("with {} maximum reflective depth", self.max_recursive_depth);
        println!("with supersampling level {}", self.supersampling_level());

        let now = std::time::Instant::now();

        let style = indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {wide_bar:.cyan/blue} pixels shaded: {human_pos}/{human_len} {percent}% ({eta})",
        )
        .unwrap();
        let pb = indicatif::ProgressBar::new(image.width() as u64 * image.height() as u64);
        let pb = pb.with_style(style);
        image.set_each_pixel(|x: usize, y: usize| self.color_at_pixel(x, y, camera), pb);
        println!("render time: {:?}", now.elapsed());
        let rays_per_sec = ray_count as f64 / now.elapsed().as_secs_f64();
        println!("rays per second: {}", rays_per_sec.round());
        image
    }

    pub fn light_sources(&self) -> &[PointLightSource] {
        self.light_sources.as_ref()
    }

    /// 0. means no shadow, 1. means full shadow
    fn point_shadow_intensity(
        &self,
        distance: f64,
        mut intersections: IntersectionCollection,
    ) -> f64 {
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
        for inter in intersections.vec_sorted() {
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

    fn get_point_shadow_dist_ray(
        &self,
        light_source: &PointLightSource,
        point: Point,
    ) -> (f64, Ray) {
        let v = light_source.position() - point;

        let distance = v.magnitude();
        let direction = v.normalize();

        let ray = Ray::new(point, direction);

        (distance, ray)
    }

    pub fn point_shadow_intensity_point(
        &self,
        light_source: &PointLightSource,
        point: Point,
    ) -> f64 {
        let (distance, ray) = self.get_point_shadow_dist_ray(light_source, point);
        let intersections = self.intersect(ray);

        self.point_shadow_intensity(distance, intersections)
    }

    pub fn point_shadow_intensity_comps(
        &self,
        light_source: &PointLightSource,
        comps: &IntersecComputations,
    ) -> f64 {
        let (dist, ray) = self.get_point_shadow_dist_ray(light_source, comps.over_point());
        let mut collector =
            IntersectionCollector::with_dest_obj_shadow_intensity(comps.object(), dist);

        self.objects.intersect(&ray, &mut collector);
        let intensity = collector.shadow_intensity().unwrap_or(0.);

        if !self.use_shadow_intensity {
            return if intensity.approx_eq(&0.) { 0. } else { 1. };
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
                let surface = light_source.color_of_illuminated_point(
                    hit_comps.object(),
                    hit_comps.over_point(),
                    hit_comps.eye_v(),
                    hit_comps.normal_v(),
                    self.point_shadow_intensity_comps(light_source, &hit_comps),
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

    pub fn use_shadow_intensity(&self) -> bool {
        self.use_shadow_intensity
    }

    pub fn set_use_shadow_intensity(&mut self, use_shadow_intensity: bool) {
        self.use_shadow_intensity = use_shadow_intensity;
    }

    pub fn max_recursive_depth(&self) -> usize {
        self.max_recursive_depth
    }

    pub fn set_max_recursive_depth(&mut self, max_recursive_depth: usize) {
        self.max_recursive_depth = max_recursive_depth;
    }

    pub fn supersampling_level(&self) -> usize {
        self.supersampling_offsets.len()
    }

    pub fn set_supersampling_level(&mut self, level: usize) {
        self.supersampling_offsets = Self::gen_supersampling_offsets(level);
    }
}

impl Scene {
    pub fn animate(&mut self, time: f64) {
        self.objects.animate(time);
    }
}

// Default testing scene with bool shadows
impl Scene {
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

    use crate::{
        assert_approx_eq_low_prec, math::vector::Vector, render::intersection::Intersection,
        Material, Pattern, Shape,
    };

    use super::*;

    #[test]
    fn intersect_scene_with_ray() {
        let scene = Scene::default_testing();
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));

        let mut intersections = scene.intersect_testing(ray);
        intersections.sort();
        assert_eq!(
            intersections.try_sorted_times_vec().unwrap(),
            vec![4., 4.5, 5.5, 6.]
        );
    }
    #[test]
    fn shade_intersection() {
        let scene = Scene::default_testing();
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));

        assert_approx_eq_low_prec!(scene.color_at(ray), Color::new(0.38066, 0.47583, 0.2855));
    }

    #[test]
    fn shade_intersection_from_inside() {
        let mut scene = Scene::default_testing();
        scene.set_light_sources(vec![PointLightSource::new(
            Point::new(0., 0.25, 0.),
            Color::new(1., 1., 1.),
        )]);

        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));

        assert_approx_eq_low_prec!(scene.color_at(ray), Color::new(0.90498, 0.90498, 0.90498));
    }

    #[test]
    fn color_when_ray_misses() {
        let scene = Scene::default_testing();
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 1., 0.));

        assert_approx_eq_low_prec!(scene.color_at(ray), Color::black());
    }

    #[test]
    fn color_when_ray_hits() {
        let scene = Scene::default_testing();
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 1., 0.));

        assert_approx_eq_low_prec!(scene.color_at(ray), Color::black());
    }

    #[test]
    fn no_shadow_when_nothing_blocks_light() {
        let scene = Scene::default_testing();
        let point = Point::new(0., 10., 0.);

        assert_approx_eq_low_prec!(
            scene.point_shadow_intensity_point(&scene.light_sources()[0], point),
            0.
        )
    }

    #[test]
    fn shadow_when_object_is_between_point_and_light() {
        let scene = Scene::default_testing();
        let point = Point::new(10., -10., 10.);

        assert_approx_eq_low_prec!(
            scene.point_shadow_intensity_point(&scene.light_sources()[0], point),
            1.
        )
    }

    #[test]
    fn no_shadow_when_object_is_behind_light() {
        let scene = Scene::default_testing();
        let point = Point::new(-20., 20., -20.);

        assert_approx_eq_low_prec!(
            scene.point_shadow_intensity_point(&scene.light_sources()[0], point),
            0.
        )
    }

    #[test]
    fn shade_hit_intersection_in_shadow() {
        let mut scene = Scene::empty();
        scene.add_light(PointLightSource::new(
            Point::new(0., 0., -10.),
            Color::white(),
        ));

        scene.add_obj(Object::primitive_with_shape(Shape::Sphere));
        scene.add_obj(Object::primitive_with_transformation(
            Shape::Sphere,
            Matrix::translation(0., 0., 10.),
        ));

        let ray = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        let inter = Intersection::new(4., &scene.objects()[1]);
        let comps = inter.computations(&ray);

        assert_approx_eq_low_prec!(scene.shade_hit(comps, 0), Color::new(0.1, 0.1, 0.1));
    }

    #[test]
    fn reflected_color_for_non_reflective_material() {
        let mut w = Scene::default_testing();
        let r = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let shape = &mut w.objects_mut()[1];
        shape.material_mut().unwrap().ambient = 1.;

        let i = Intersection::new(1., &w.objects()[1]);
        let comps = i.computations(&r);

        assert_approx_eq_low_prec!(w.reflected_color(&comps, 0), Color::black());
    }

    #[test]
    fn shade_hit_with_reflective_material() {
        let mut w = Scene::default_testing();
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
        let mut w = Scene::empty();

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
        let mut scene = Scene::default_testing();
        let plane = Object::primitive(
            Shape::Plane,
            Material {
                reflectivity: 0.5,
                ..Default::default()
            },
            Matrix::translation(0., -1., 0.),
        );
        scene.add_obj(plane);

        let r = Ray::new(
            Point::new(0., 0., -3.),
            Vector::new(0., -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        );
        let i = Intersection::new(SQRT_2, scene.objects().last().unwrap());
        let comps = i.computations(&r);

        assert_approx_eq_low_prec!(
            scene.reflected_color(&comps, scene.max_recursive_depth),
            Color::black()
        );
    }

    #[test]
    fn refraced_colr_with_opaque_surface() {
        let scene = Scene::default_testing();
        let shape = &scene.objects()[0];
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let intersections = IntersectionCollection::from_times_and_obj(ray, vec![4., 6.], shape);
        let comps = intersections.hit_computations().unwrap();

        assert_approx_eq_low_prec!(scene.refracted_color(&comps, 0), Color::black());
    }

    #[test]
    fn refracted_color_at_max_recursive_depth() {
        let mut scene = Scene::default_testing();
        let shape = &mut scene.objects_mut()[0];
        shape.material_mut().unwrap().transparency = 1.;
        shape.material_mut().unwrap().refractive_index = 1.5;
        let shape = &scene.objects()[0];

        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let intersections = IntersectionCollection::from_times_and_obj(ray, vec![4., 6.], shape);
        let comps = intersections.hit_computations().unwrap();

        assert_approx_eq_low_prec!(
            scene.refracted_color(&comps, scene.max_recursive_depth),
            Color::black()
        );
    }

    #[test]
    fn refracted_color_under_total_internal_reflection() {
        let mut scene = Scene::default_testing();
        let shape = &mut scene.objects_mut()[0];
        shape.material_mut().unwrap().transparency = 1.;
        shape.material_mut().unwrap().refractive_index = 1.5;
        let shape = &scene.objects()[0];

        let ray = Ray::new(Point::new(0., 0., SQRT_2 / 2.), Vector::new(0., 1., 0.));
        let intersections =
            IntersectionCollection::from_times_and_obj(ray, vec![-SQRT_2 / 2., SQRT_2 / 2.], shape);

        let comps = intersections.computations_at_id(1).unwrap();

        assert_approx_eq_low_prec!(scene.refracted_color(&comps, 0), Color::black());
    }

    #[test]
    fn refracted_color_with_refracted_ray() {
        let mut scene = Scene::default_testing();

        let a = &mut scene.objects_mut()[0];
        a.material_mut().unwrap().ambient = 1.;
        a.material_mut().unwrap().pattern = Pattern::test_pattern(None);

        let b = &mut scene.objects_mut()[1];
        b.material_mut().unwrap().transparency = 1.;
        b.material_mut().unwrap().refractive_index = 1.5;

        let ray = Ray::new(Point::new(0., 0., 0.1), Vector::new(0., 1., 0.));

        let a = &scene.objects()[0];
        let b = &scene.objects()[1];
        let objects = vec![a.clone(), b.clone()];

        let intersections = IntersectionCollection::from_ray_and_mult_objects(ray, &objects);
        let comps = intersections.hit_computations().unwrap();

        assert_approx_eq_low_prec!(
            scene.refracted_color(&comps, 0),
            &Color::new(0., 0.99888, 0.04725)
        );
    }

    #[test]
    fn shading_transparent_material() {
        let mut scene = Scene::default_testing();
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
        scene.add_obj(floor);
        scene.add_obj(ball);

        let ray = Ray::new(
            Point::new(0., 0., -3.),
            Vector::new(0., -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        );
        let intersections = scene.intersect(ray);
        let cmps = intersections.hit_computations().unwrap();

        assert_approx_eq_low_prec!(
            scene.shade_hit(cmps, 0),
            Color::new(0.93642, 0.68642, 0.68642)
        );
    }

    #[test]
    fn shading_reflective_transparent_material() {
        let mut scene = Scene::default_testing();
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
        scene.add_obj(floor);
        scene.add_obj(ball);

        let ray = Ray::new(
            Point::new(0., 0., -3.),
            Vector::new(0., -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        );
        let intersections = scene.intersect(ray);
        let cmps = intersections.hit_computations().unwrap();

        assert_approx_eq_low_prec!(
            scene.shade_hit(cmps, 0),
            Color::new(0.93391, 0.69643, 0.69243)
        );
    }

    #[test]
    fn default_scene_builder() {
        let builder = SceneBuilder::default();
        builder.build().unwrap();
    }

    #[test]
    fn default_scene_values() {
        let builded = SceneBuilder::default().build().unwrap();
        let scene = Scene::default();

        assert_eq!(
            builded.supersampling_level(),
            Scene::DEFAULT_SUPERSAMPLING_LEVEL
        );
        assert_eq!(
            scene.supersampling_level(),
            Scene::DEFAULT_SUPERSAMPLING_LEVEL
        );

        assert_eq!(scene.max_recursive_depth(), Scene::MAX_RECURSIVE_DEPTH);
        assert_eq!(builded.max_recursive_depth(), Scene::MAX_RECURSIVE_DEPTH);

        assert!(scene.use_shadow_intensity());
        assert!(builded.use_shadow_intensity());
    }
}
