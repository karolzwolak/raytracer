use derive_builder::Builder;

use crate::{
    math::{approx_eq::ApproxEq, color::Color, point::Point},
    render::ray::{
        intersection::{IntersecComputations, IntersectionCollection, IntersectionCollector},
        Ray,
    },
    scene::{
        light::{point_light::PointLightSource, schlick_reflectance},
        object::group::ObjectGroup,
        Scene,
    },
};

#[derive(PartialEq, Debug, Clone, Builder)]
#[builder(default)]
/// The integrator calculates the color for each given ray
pub struct Integrator {
    /// The scene to be rendered
    scene: Scene,
    /// Depth of recursive calls for reflections and refractions
    /// 0 means no reflections or refractions
    #[builder(default = "Integrator::MAX_RECURSIVE_DEPTH")]
    max_recursive_depth: usize,
    /// If true, shadows are calculated with intensity,
    /// so that all objects don't cast full shadow
    /// boolean shadows are required for testing purposes,
    /// because all tests values were calculated with bool shadows
    #[builder(default = "true")]
    use_shadow_intensity: bool,
}

impl Default for Integrator {
    fn default() -> Self {
        Self {
            scene: Scene::default(),
            max_recursive_depth: Self::MAX_RECURSIVE_DEPTH,
            use_shadow_intensity: true,
        }
    }
}

impl Integrator {
    const MAX_RECURSIVE_DEPTH: usize = 5 - 1;

    fn objects(&self) -> &ObjectGroup {
        self.scene.objects()
    }

    fn light_sources(&self) -> &[PointLightSource] {
        self.scene.light_sources()
    }

    fn intersect(&self, ray: Ray) -> IntersectionCollection {
        IntersectionCollection::from_group(ray, self.objects())
    }

    fn color_at_depth(&self, ray: Ray, depth: usize) -> Color {
        self.intersect(ray)
            .hit_computations()
            .map_or(Color::black(), |hit_comps| {
                self.shade_hit_at_depth(hit_comps, depth)
            })
    }

    /// The main method for calculating color for the given ray
    pub fn color_at(&self, ray: Ray) -> Color {
        self.color_at_depth(ray, 0)
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

    fn point_shadow_intensity_comps(
        &self,
        light_source: &PointLightSource,
        comps: &IntersecComputations,
    ) -> f64 {
        let (dist, ray) = self.get_point_shadow_dist_ray(light_source, comps.over_point());
        let mut collector =
            IntersectionCollector::with_dest_obj_shadow_intensity(comps.object(), dist);

        self.objects().intersect(&ray, &mut collector);
        let intensity = collector.shadow_intensity().unwrap_or(0.);

        if !self.use_shadow_intensity {
            return if intensity.approx_eq(&0.) { 0. } else { 1. };
        }
        intensity
    }

    fn reflected_color_at_depth(&self, hit_comps: &IntersecComputations, depth: usize) -> Color {
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

    fn refracted_color_at_depth(&self, hit_comps: &IntersecComputations, depth: usize) -> Color {
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

    fn shade_hit_at_depth(&self, hit_comps: IntersecComputations, depth: usize) -> Color {
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
                let reflected = self.reflected_color_at_depth(&hit_comps, depth);
                let refracted = self.refracted_color_at_depth(&hit_comps, depth);

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

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }
}

impl Integrator {
    #[cfg(test)]
    fn refracted_color(&self, hit_comps: &IntersecComputations) -> Color {
        self.refracted_color_at_depth(hit_comps, 0)
    }

    #[cfg(test)]
    fn reflected_color(&self, hit_comps: &IntersecComputations) -> Color {
        self.reflected_color_at_depth(hit_comps, 0)
    }

    #[cfg(test)]
    fn shade_hit(&self, hit_comps: IntersecComputations) -> Color {
        self.shade_hit_at_depth(hit_comps, 0)
    }

    #[cfg(test)]
    fn intersect_testing(&self, ray: Ray) -> IntersectionCollection {
        let mut collector = IntersectionCollector::new_keep_redundant();
        self.objects().intersect(&ray, &mut collector);
        IntersectionCollection::from_collector(ray, collector)
    }

    #[cfg(test)]
    pub fn default_testing(scene: Scene) -> Self {
        Self {
            scene,
            max_recursive_depth: Self::MAX_RECURSIVE_DEPTH,
            use_shadow_intensity: false,
        }
    }
}

impl IntegratorBuilder {
    #[cfg(test)]
    pub fn default_builder_testing() -> Self {
        let mut builder = IntegratorBuilder::default();
        builder
            .scene(Scene::default_testing())
            .use_shadow_intensity(false);
        builder
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        math::{matrix::Matrix, tuple::Tuple},
        scene::object::{
            material::{pattern::Pattern, Material},
            primitive::shape::Shape,
            Object,
        },
    };
    use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

    use super::*;
    use crate::{
        assert_approx_eq_low_prec, math::vector::Vector, render::ray::intersection::Intersection,
        scene::SceneBuilder,
    };

    #[test]
    fn intersect_integrator_with_ray() {
        let integrator = Integrator::default_testing(Scene::default_testing());
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));

        let mut intersections = integrator.intersect_testing(ray);
        intersections.sort();
        assert_eq!(
            intersections.try_sorted_times_vec().unwrap(),
            vec![4., 4.5, 5.5, 6.]
        );
    }

    fn test_color_at(scene: Scene, ray: Ray, expected_color: Color) {
        let integrator = Integrator::default_testing(scene);

        assert_approx_eq_low_prec!(integrator.color_at(ray), expected_color);
    }

    fn test_shadow_intensity_at_point(point: Point, expected_intensity: f64) {
        let integrator = Integrator::default_testing(Scene::default_testing());

        assert_approx_eq_low_prec!(
            integrator.point_shadow_intensity_point(&integrator.light_sources()[0], point),
            expected_intensity
        );
    }
    #[test]
    fn shade_intersection() {
        test_color_at(
            Scene::default_testing(),
            Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.)),
            Color::new(0.38066, 0.47583, 0.2855),
        );
    }

    #[test]
    fn shade_intersection_from_inside() {
        let mut scene = SceneBuilder::default_builder_testing();
        scene.light_sources(vec![PointLightSource::new(
            Point::new(0., 0.25, 0.),
            Color::new(1., 1., 1.),
        )]);
        test_color_at(
            scene.clone().build(),
            Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.)),
            Color::new(0.90498, 0.90498, 0.90498),
        );
    }

    #[test]
    fn color_when_ray_misses() {
        test_color_at(
            Scene::default_testing(),
            Ray::new(Point::new(0., 0., -5.), Vector::new(0., 1., 0.)),
            Color::black(),
        );
    }

    #[test]
    fn color_when_ray_hits() {
        test_color_at(
            Scene::default_testing(),
            Ray::new(Point::new(0., 0., -5.), Vector::new(0., 1., 0.)),
            Color::black(),
        );
    }

    #[test]
    fn no_shadow_when_nothing_blocks_light() {
        test_shadow_intensity_at_point(Point::new(0., 10., 0.), 0.);
    }

    #[test]
    fn shadow_when_object_is_between_point_and_light() {
        test_shadow_intensity_at_point(Point::new(10., -10., 10.), 1.);
    }

    #[test]
    fn no_shadow_when_object_is_behind_light() {
        test_shadow_intensity_at_point(Point::new(-20., 20., -20.), 0.);
    }

    #[test]
    fn reflected_color_for_non_reflective_material() {
        let mut scene = SceneBuilder::default_builder_testing();
        let r = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let shape = &mut scene.get_objects_mut().children_mut()[1];
        shape.material_mut().unwrap().ambient = 1.;

        let integrator = Integrator::default_testing(scene.build());

        let i = Intersection::new(1., &integrator.objects().children()[1]);
        let comps = i.computations(&r);

        assert_approx_eq_low_prec!(integrator.reflected_color(&comps), Color::black());
    }

    #[test]
    fn shade_hit_intersection_in_shadow() {
        let mut scene = SceneBuilder::default_builder_testing();
        scene.light_sources(vec![PointLightSource::new(
            Point::new(0., 0., -10.),
            Color::white(),
        )]);

        scene.objects(vec![
            Object::primitive_with_shape(Shape::Sphere),
            Object::primitive_with_transformation(Shape::Sphere, Matrix::translation(0., 0., 10.)),
        ]);

        let integrator = Integrator::default_testing(scene.build());

        let ray = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        let inter = Intersection::new(4., &integrator.objects().children()[1]);
        let comps = inter.computations(&ray);

        assert_approx_eq_low_prec!(integrator.shade_hit(comps), Color::new(0.1, 0.1, 0.1));
    }

    #[test]
    fn shade_hit_with_reflective_material() {
        let mut scene = SceneBuilder::default_builder_testing();
        let plane = Object::primitive(
            Shape::Plane,
            Material {
                reflectivity: 0.5,
                ..Default::default()
            },
            Matrix::translation(0., -1., 0.),
        );
        scene.add_object(plane);
        let integrator = Integrator::default_testing(scene.build());

        let r = Ray::new(
            Point::new(0., 0., -3.),
            Vector::new(0., -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        );
        let i = Intersection::new(SQRT_2, integrator.objects().children().last().unwrap());
        let comps = i.computations(&r);

        assert!(integrator
            .shade_hit(comps)
            .approx_eq_low_prec(&Color::new(0.87677, 0.92436, 0.82918)));
    }

    #[test]
    fn ray_bouncing_between_mutually_reflective_surfaces_doesnt_crash() {
        let mut scene = SceneBuilder::default();

        scene.add_light_source(PointLightSource::new(
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
        scene.add_object(lower);
        scene.add_object(upper);
        let scene = scene.build();

        let integrator = IntegratorBuilder::default().scene(scene).build().unwrap();

        let r = Ray::new(Point::zero(), Vector::new(0., 1., 0.));

        let _ = integrator.color_at(r);
    }

    #[test]
    fn reflected_color_at_max_recursive_depth() {
        let mut scene = SceneBuilder::default_builder_testing();
        let plane = Object::primitive(
            Shape::Plane,
            Material {
                reflectivity: 0.5,
                ..Default::default()
            },
            Matrix::translation(0., -1., 0.),
        );
        scene.add_object(plane);
        let integrator = Integrator::default_testing(scene.build());

        let r = Ray::new(
            Point::new(0., 0., -3.),
            Vector::new(0., -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        );
        let i = Intersection::new(SQRT_2, integrator.objects().children().last().unwrap());
        let comps = i.computations(&r);

        assert_approx_eq_low_prec!(
            integrator.reflected_color_at_depth(&comps, integrator.max_recursive_depth),
            Color::black()
        );
    }

    #[test]
    fn refraced_colr_with_opaque_surface() {
        let integrator = Integrator::default_testing(Scene::default_testing());
        let shape = &integrator.objects().children()[0];
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let intersections = IntersectionCollection::from_times_and_obj(ray, vec![4., 6.], shape);
        let comps = intersections.hit_computations().unwrap();

        assert_approx_eq_low_prec!(integrator.refracted_color(&comps), Color::black());
    }

    #[test]
    fn refracted_color_at_max_recursive_depth() {
        let mut scene = SceneBuilder::default_builder_testing();
        let shape = &mut scene.get_objects_mut().children_mut()[0];
        shape.material_mut().unwrap().transparency = 1.;
        shape.material_mut().unwrap().refractive_index = 1.5;

        let integrator = Integrator::default_testing(scene.build());
        let shape = &integrator.objects().children()[0];

        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let intersections = IntersectionCollection::from_times_and_obj(ray, vec![4., 6.], shape);
        let comps = intersections.hit_computations().unwrap();

        assert_approx_eq_low_prec!(
            integrator.refracted_color_at_depth(&comps, integrator.max_recursive_depth),
            Color::black()
        );
    }

    #[test]
    fn refracted_color_under_total_internal_reflection() {
        let mut scene = SceneBuilder::default_builder_testing();
        let shape = &mut scene.get_objects_mut().children_mut()[0];
        shape.material_mut().unwrap().transparency = 1.;
        shape.material_mut().unwrap().refractive_index = 1.5;

        let integrator = Integrator::default_testing(scene.build());
        let shape = &integrator.objects().children()[0];

        let ray = Ray::new(Point::new(0., 0., SQRT_2 / 2.), Vector::new(0., 1., 0.));
        let intersections =
            IntersectionCollection::from_times_and_obj(ray, vec![-SQRT_2 / 2., SQRT_2 / 2.], shape);

        let comps = intersections.computations_at_id(1).unwrap();

        assert_approx_eq_low_prec!(integrator.refracted_color(&comps), Color::black());
    }

    #[test]
    fn refracted_color_with_refracted_ray() {
        let mut scene = SceneBuilder::default_builder_testing();

        let a = &mut scene.get_objects_mut().children_mut()[0];
        a.material_mut().unwrap().ambient = 1.;
        a.material_mut().unwrap().pattern = Pattern::test_pattern(None);

        let b = &mut scene.get_objects_mut().children_mut()[1];
        b.material_mut().unwrap().transparency = 1.;
        b.material_mut().unwrap().refractive_index = 1.5;

        let integrator = Integrator::default_testing(scene.build());

        let ray = Ray::new(Point::new(0., 0., 0.1), Vector::new(0., 1., 0.));

        let a = &integrator.objects().children()[0];
        let b = &integrator.objects().children()[1];
        let objects = vec![a.clone(), b.clone()];

        let intersections = IntersectionCollection::from_ray_and_mult_objects(ray, &objects);
        let comps = intersections.hit_computations().unwrap();

        assert_approx_eq_low_prec!(
            integrator.refracted_color(&comps),
            &Color::new(0., 0.99888, 0.04725)
        );
    }

    #[test]
    fn shading_transparent_material() {
        let mut scene = SceneBuilder::default_builder_testing();
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
        scene.add_object(floor);
        scene.add_object(ball);

        let integrator = Integrator::default_testing(scene.build());

        let ray = Ray::new(
            Point::new(0., 0., -3.),
            Vector::new(0., -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        );
        let intersections = integrator.intersect(ray);
        let cmps = intersections.hit_computations().unwrap();

        assert_approx_eq_low_prec!(
            integrator.shade_hit(cmps),
            Color::new(0.93642, 0.68642, 0.68642)
        );
    }

    #[test]
    fn shading_reflective_transparent_material() {
        let mut scene = SceneBuilder::default_builder_testing();
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
        scene.add_object(floor);
        scene.add_object(ball);

        let integrator = Integrator::default_testing(scene.build());

        let ray = Ray::new(
            Point::new(0., 0., -3.),
            Vector::new(0., -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        );
        let intersections = integrator.intersect(ray);
        let cmps = intersections.hit_computations().unwrap();

        assert_approx_eq_low_prec!(
            integrator.shade_hit(cmps),
            Color::new(0.93391, 0.69643, 0.69243)
        );
    }

    #[test]
    fn default_integrator_builder() {
        let builder = IntegratorBuilder::default();
        builder.build().unwrap();
    }

    #[test]
    fn default_integrator_values() {
        let builded = IntegratorBuilder::default().build().unwrap();
        let default = Integrator::default();

        let expected = Integrator {
            scene: Scene::default(),
            max_recursive_depth: Integrator::MAX_RECURSIVE_DEPTH,
            use_shadow_intensity: true,
        };

        assert_eq!(builded, expected);
        assert_eq!(default, expected);
    }
}
