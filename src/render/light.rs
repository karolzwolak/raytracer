use crate::primitive::{point::Point, vector::Vector};

use super::{color::Color, material::Material};

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
pub fn lighting(
    material: &Material,
    light: &PointLightSource,
    point: Point,
    eye_v: Vector,
    normal_v: Vector,
) -> Color {
    // combine surface color with lights's intensity (color)
    let effetive_color = material.color() * light.intensity;

    // direction to the light source
    let light_v = (light.position() - point).normalize();

    let ambient = effetive_color * material.ambient();

    let light_dot_normal = light_v.dot(normal_v);

    // if cosine between light and normal vectors is negative, light is on the other side of surface
    if light_dot_normal < 0. {
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
            light.intensity * material.specular() * factor
        }
    };

    ambient + diffuse + specular
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use crate::primitive::tuple::Tuple;

    use super::*;

    #[test]
    fn lighting_with_eye_between_light_and_surface() {
        let point = Point::zero();
        let material = Material::default();

        let eye_v = Vector::new(0., 0., -1.);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 0., -10.), Color::white());

        assert_eq!(
            lighting(&material, &light, point, eye_v, normal_v),
            Color::new(1.9, 1.9, 1.9)
        );
    }
    #[test]
    fn lighting_with_eye_between_light_and_surface_eye_offset_45() {
        let point = Point::zero();
        let material = Material::default();

        let eye_v = Vector::new(0., FRAC_1_SQRT_2, -FRAC_1_SQRT_2);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 0., -10.), Color::white());

        assert_eq!(
            lighting(&material, &light, point, eye_v, normal_v),
            Color::new(1.0, 1.0, 1.0)
        );
    }
    #[test]
    fn lighting_with_eye_opposite_surface_light_offset_45() {
        let point = Point::zero();
        let material = Material::default();

        let eye_v = Vector::new(0., 0., -1.);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 10., -10.), Color::white());

        let intensity = 0.1 + 0.9 * FRAC_1_SQRT_2;
        assert_eq!(
            lighting(&material, &light, point, eye_v, normal_v),
            Color::new(intensity, intensity, intensity)
        );
    }
    #[test]
    fn lighting_with_eye_in_path_of_reflection() {
        let point = Point::zero();
        let material = Material::default();

        let eye_v = Vector::new(0., -FRAC_1_SQRT_2, -FRAC_1_SQRT_2);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 10., -10.), Color::white());

        let intensity = 1. + 0.9 * FRAC_1_SQRT_2;
        assert_eq!(
            lighting(&material, &light, point, eye_v, normal_v),
            Color::new(intensity, intensity, intensity)
        );
    }
    #[test]
    fn lighting_with_light_behind_surface() {
        let point = Point::zero();
        let material = Material::default();

        let eye_v = Vector::new(0., 0., -1.);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 0., 10.), Color::white());

        assert_eq!(
            lighting(&material, &light, point, eye_v, normal_v),
            Color::new(0.1, 0.1, 0.1)
        );
    }
}
