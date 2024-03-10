use crate::{
    approx_eq::ApproxEq,
    primitive::{point::Point, vector::Vector},
};

use super::intersection::IntersecComputations;
use super::{color::Color, object::Object};

pub struct PointLightSource {
    position: Point,
    intensity: Color,
}

impl PointLightSource {
    pub fn new(position: Point, intensity: Color) -> Self {
        Self {
            position,
            intensity,
        }
    }

    pub fn intensity(&self) -> Color {
        self.intensity
    }

    pub fn position(&self) -> Point {
        self.position
    }
}

/// compute color of illuminated point using Phong reflection model
pub fn color_of_illuminated_point(
    object: &Object,
    light_source: &PointLightSource,
    point: Point,
    eye_v: Vector,
    normal_v: Vector,
    shadow_intensity: f64,
) -> Color {
    let material = object.material();
    // combine surface color with lights's intensity (color)
    let effetive_color = material.color_at_object(object, point) * light_source.intensity;

    // direction to the light source
    let light_v = (light_source.position() - point).normalize();

    let ambient = effetive_color * material.ambient;

    let light_dot_normal = light_v.dot(normal_v);

    // if cosine between light and normal vectors is negative, light is on the other side of surface
    if shadow_intensity.approx_eq(&1.) || light_dot_normal < 0. {
        return ambient + Color::black() + Color::black();
    }
    let diffuse = effetive_color * material.diffuse * light_dot_normal;

    let reflect_v = (-light_v).reflect(normal_v);
    let reflect_dot_eye = reflect_v.dot(eye_v);

    // if cosine between reflect and eye vectors is negative, light reflects away from the eye
    let specular = match reflect_dot_eye.is_sign_positive() {
        false => Color::black(),
        true => {
            let factor = reflect_dot_eye.powf(material.shininess);
            light_source.intensity * material.specular * factor
        }
    };

    ambient + (diffuse + specular) * (1. - shadow_intensity)
}

pub fn schlick_reflectance(comps: &IntersecComputations) -> f64 {
    // by default use cos of theta_i
    let mut cos = comps.eye_v().dot(comps.normal_v());

    if comps.refractive_from() > comps.refractive_to() {
        let ratio = comps.refractive_from() / comps.refractive_to();
        let sin2_t = ratio.powi(2) * (1. - cos.powi(2));
        if sin2_t > 1. {
            return 1.;
        }

        let cos_t = (1. - sin2_t).sqrt();

        // when reflective from greater than refractive to, use cos of theta_t
        cos = cos_t;
    }

    let r0 = ((comps.refractive_from() - comps.refractive_to())
        / (comps.refractive_from() + comps.refractive_to()))
    .powi(2);

    r0 + (1. - r0) * (1. - cos).powi(5)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use crate::{
        approx_eq::ApproxEq,
        primitive::{matrix::Matrix, tuple::Tuple},
        render::{
            intersection::IntersectionCollection,
            material::Material,
            object::{shape::Shape, PrimitiveObject},
            pattern::Pattern,
            ray::Ray,
        },
    };

    use super::*;

    #[test]
    fn lighting_with_surface_in_shadow() {
        let point = Point::zero();
        let obj = PrimitiveObject::with_shape(Shape::Sphere).into();

        let eye_v = Vector::new(0., 0., -1.);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 0., -10.), Color::white());

        assert_eq!(
            color_of_illuminated_point(&obj, &light, point, eye_v, normal_v, 1.),
            Color::new(0.1, 0.1, 0.1)
        );
    }
    #[test]
    fn lighting_with_eye_between_light_and_surface() {
        let point = Point::zero();
        let obj = PrimitiveObject::with_shape(Shape::Sphere).into();

        let eye_v = Vector::new(0., 0., -1.);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 0., -10.), Color::white());

        assert_eq!(
            color_of_illuminated_point(&obj, &light, point, eye_v, normal_v, 0.),
            Color::new(1.9, 1.9, 1.9)
        );
    }
    #[test]
    fn lighting_with_eye_between_light_and_surface_eye_offset_45() {
        let point = Point::zero();
        let obj = PrimitiveObject::with_shape(Shape::Sphere).into();

        let eye_v = Vector::new(0., FRAC_1_SQRT_2, -FRAC_1_SQRT_2);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 0., -10.), Color::white());

        assert_eq!(
            color_of_illuminated_point(&obj, &light, point, eye_v, normal_v, 0.),
            Color::new(1.0, 1.0, 1.0)
        );
    }
    #[test]
    fn lighting_with_eye_opposite_surface_light_offset_45() {
        let point = Point::zero();
        let obj = PrimitiveObject::with_shape(Shape::Sphere).into();

        let eye_v = Vector::new(0., 0., -1.);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 10., -10.), Color::white());

        let intensity = 0.1 + 0.9 * FRAC_1_SQRT_2;
        assert_eq!(
            color_of_illuminated_point(&obj, &light, point, eye_v, normal_v, 0.),
            Color::new(intensity, intensity, intensity)
        );
    }
    #[test]
    fn lighting_with_eye_in_path_of_reflection() {
        let point = Point::zero();
        let obj = PrimitiveObject::with_shape(Shape::Sphere).into();

        let eye_v = Vector::new(0., -FRAC_1_SQRT_2, -FRAC_1_SQRT_2);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 10., -10.), Color::white());

        let intensity = 1. + 0.9 * FRAC_1_SQRT_2;
        assert_eq!(
            color_of_illuminated_point(&obj, &light, point, eye_v, normal_v, 0.),
            Color::new(intensity, intensity, intensity)
        );
    }
    #[test]
    fn lighting_with_light_behind_surface() {
        let point = Point::zero();
        let obj = PrimitiveObject::with_shape(Shape::Sphere).into();

        let eye_v = Vector::new(0., 0., -1.);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 0., 10.), Color::white());

        assert_eq!(
            color_of_illuminated_point(&obj, &light, point, eye_v, normal_v, 0.),
            Color::new(0.1, 0.1, 0.1)
        );
    }
    #[test]
    fn lighting_with_pattern_applied() {
        let material = Material {
            pattern: Pattern::stripe(Color::white(), Color::black(), None),
            ambient: 1.,
            diffuse: 0.,
            specular: 0.,
            ..Default::default()
        };
        let obj = PrimitiveObject::with_shape_material(Shape::Sphere, material).into();

        let eye_v = Vector::new(0., 0., -1.);
        let normal_v = Vector::new(0., 0., -1.);
        let light_source = PointLightSource::new(Point::new(0., 0., -10.), Color::white());

        assert_eq!(
            color_of_illuminated_point(
                &obj,
                &light_source,
                Point::new(0.9, 0., 0.),
                eye_v,
                normal_v,
                0.
            ),
            Color::white()
        );
        assert_eq!(
            color_of_illuminated_point(
                &obj,
                &light_source,
                Point::new(1.1, 0., 0.),
                eye_v,
                normal_v,
                0.
            ),
            Color::black()
        );
    }

    #[test]
    fn schlick_reflectance_under_total_internal_reflection() {
        let sphere = Object::primitive(Shape::Sphere, Material::glass(), Matrix::identity());

        let ray = Ray::new(Point::new(0., 0., FRAC_1_SQRT_2), Vector::new(0., 1., 0.));
        let intersections = IntersectionCollection::from_times_and_obj(
            ray,
            vec![-FRAC_1_SQRT_2, FRAC_1_SQRT_2],
            &sphere,
        );
        let comps = intersections.hit_computations().unwrap();

        assert_eq!(schlick_reflectance(&comps), 1.);
    }

    #[test]
    fn schlick_reflectance_with_perpendicular_viewing_angle() {
        let sphere = Object::primitive(Shape::Sphere, Material::glass(), Matrix::identity());

        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 1., 0.));
        let intersections = IntersectionCollection::from_times_and_obj(ray, vec![-1., 1.], &sphere);
        let comps = intersections.hit_computations().unwrap();

        assert!(schlick_reflectance(&comps).approx_eq(&0.04));
    }

    #[test]
    fn schlick_reflectance_with_small_angle_and_n1_greater_than_n2() {
        let sphere = Object::primitive(Shape::Sphere, Material::glass(), Matrix::identity());

        let ray = Ray::new(Point::new(0., 0.99, -2.), Vector::new(0., 0., 1.));
        let intersections = IntersectionCollection::from_times_and_obj(ray, vec![1.8589], &sphere);
        let comps = intersections.hit_computations().unwrap();

        assert!(schlick_reflectance(&comps).approx_eq(&0.48873));
    }
}
