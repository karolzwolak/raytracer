use crate::primitive::{point::Point, vector::Vector};

use super::{color::Color, material::Material, object::Object};

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
    in_shadow: bool,
) -> Color {
    let material = object.material();
    // combine surface color with lights's intensity (color)
    let effetive_color = material.color_at_object(object, point) * light_source.intensity;

    // direction to the light source
    let light_v = (light_source.position() - point).normalize();

    let ambient = effetive_color * material.ambient();

    let light_dot_normal = light_v.dot(normal_v);

    // if cosine between light and normal vectors is negative, light is on the other side of surface
    if in_shadow || light_dot_normal < 0. {
        return ambient + Color::black() + Color::black();
    }
    let diffuse = effetive_color * material.diffuse() * light_dot_normal;

    let reflect_v = (-light_v).reflect(normal_v);
    let reflect_dot_eye = reflect_v.dot(eye_v);

    // if cosine between reflect and eye vectors is negative, light reflects away from the eye
    let specular = match reflect_dot_eye.is_sign_positive() {
        false => Color::black(),
        true => {
            let factor = reflect_dot_eye.powf(material.shininess());
            light_source.intensity * material.specular() * factor
        }
    };

    ambient + diffuse + specular
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use crate::{
        primitive::tuple::Tuple,
        render::{
            object::Shape::{self},
            pattern::Pattern,
        },
    };

    use super::*;

    #[test]
    fn lighting_with_surface_in_shadow() {
        let point = Point::zero();
        let obj = Object::with_shape(Shape::Sphere);

        let eye_v = Vector::new(0., 0., -1.);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 0., -10.), Color::white());

        assert_eq!(
            color_of_illuminated_point(&obj, &light, point, eye_v, normal_v, true),
            Color::new(0.1, 0.1, 0.1)
        );
    }
    #[test]
    fn lighting_with_eye_between_light_and_surface() {
        let point = Point::zero();
        let obj = Object::with_shape(Shape::Sphere);

        let eye_v = Vector::new(0., 0., -1.);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 0., -10.), Color::white());

        assert_eq!(
            color_of_illuminated_point(&obj, &light, point, eye_v, normal_v, false),
            Color::new(1.9, 1.9, 1.9)
        );
    }
    #[test]
    fn lighting_with_eye_between_light_and_surface_eye_offset_45() {
        let point = Point::zero();
        let obj = Object::with_shape(Shape::Sphere);

        let eye_v = Vector::new(0., FRAC_1_SQRT_2, -FRAC_1_SQRT_2);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 0., -10.), Color::white());

        assert_eq!(
            color_of_illuminated_point(&obj, &light, point, eye_v, normal_v, false),
            Color::new(1.0, 1.0, 1.0)
        );
    }
    #[test]
    fn lighting_with_eye_opposite_surface_light_offset_45() {
        let point = Point::zero();
        let obj = Object::with_shape(Shape::Sphere);

        let eye_v = Vector::new(0., 0., -1.);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 10., -10.), Color::white());

        let intensity = 0.1 + 0.9 * FRAC_1_SQRT_2;
        assert_eq!(
            color_of_illuminated_point(&obj, &light, point, eye_v, normal_v, false),
            Color::new(intensity, intensity, intensity)
        );
    }
    #[test]
    fn lighting_with_eye_in_path_of_reflection() {
        let point = Point::zero();
        let obj = Object::with_shape(Shape::Sphere);

        let eye_v = Vector::new(0., -FRAC_1_SQRT_2, -FRAC_1_SQRT_2);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 10., -10.), Color::white());

        let intensity = 1. + 0.9 * FRAC_1_SQRT_2;
        assert_eq!(
            color_of_illuminated_point(&obj, &light, point, eye_v, normal_v, false),
            Color::new(intensity, intensity, intensity)
        );
    }
    #[test]
    fn lighting_with_light_behind_surface() {
        let point = Point::zero();
        let obj = Object::with_shape(Shape::Sphere);

        let eye_v = Vector::new(0., 0., -1.);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 0., 10.), Color::white());

        assert_eq!(
            color_of_illuminated_point(&obj, &light, point, eye_v, normal_v, false),
            Color::new(0.1, 0.1, 0.1)
        );
    }
    #[test]
    fn lighting_with_pattern_applied() {
        let mut material =
            Material::with_pattern(Pattern::stripe(Color::white(), Color::black(), None));
        material.set_ambient(1.);
        material.set_diffuse(0.);
        material.set_specular(0.);
        let obj = Object::with_shape_material(Shape::Sphere, material);

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
                false
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
                false
            ),
            Color::black()
        );
    }
}
