use crate::{
    math::{approx_eq::ApproxEq, color::Color, point::Point, vector::Vector},
    scene::Object,
};

#[derive(PartialEq, Debug, Clone)]
pub struct PointLightSource {
    position: Point,
    intensity: Color,
}

impl Default for PointLightSource {
    fn default() -> Self {
        Self {
            position: Point::zero(),
            intensity: Color::white(),
        }
    }
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

    /// compute color of illuminated point using Phong reflection model
    pub fn color_of_illuminated_point(
        &self,
        object: &Object,
        point: Point,
        eye_v: Vector,
        normal_v: Vector,
        shadow_intensity: f64,
    ) -> Color {
        let material = object.material_unwrapped();
        // combine surface color with lights's intensity (color)
        let effetive_color = material.color_at_object(object, point) * self.intensity;

        // direction to the light source
        let light_v = (self.position() - point).normalize();

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
                self.intensity * material.specular * factor
            }
        };

        ambient + (diffuse + specular) * (1. - shadow_intensity)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use super::*;
    use crate::{
        assert_approx_eq_low_prec,
        math::tuple::Tuple,
        scene::object::{
            PrimitiveObject,
            material::{Material, pattern::Pattern},
            primitive::shape::Shape,
        },
    };

    #[test]
    fn lighting_with_surface_in_shadow() {
        let point = Point::zero();
        let obj = PrimitiveObject::with_shape(Shape::Sphere).into();

        let eye_v = Vector::new(0., 0., -1.);
        let normal_v = Vector::new(0., 0., -1.);
        let light = PointLightSource::new(Point::new(0., 0., -10.), Color::white());

        assert_approx_eq_low_prec!(
            light.color_of_illuminated_point(&obj, point, eye_v, normal_v, 1.),
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

        assert_approx_eq_low_prec!(
            light.color_of_illuminated_point(&obj, point, eye_v, normal_v, 0.),
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

        assert_approx_eq_low_prec!(
            light.color_of_illuminated_point(&obj, point, eye_v, normal_v, 0.),
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
        assert_approx_eq_low_prec!(
            light.color_of_illuminated_point(&obj, point, eye_v, normal_v, 0.),
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
        assert_approx_eq_low_prec!(
            light.color_of_illuminated_point(&obj, point, eye_v, normal_v, 0.),
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

        assert_approx_eq_low_prec!(
            light.color_of_illuminated_point(&obj, point, eye_v, normal_v, 0.),
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
        let light = PointLightSource::new(Point::new(0., 0., -10.), Color::white());

        assert_approx_eq_low_prec!(
            light.color_of_illuminated_point(&obj, Point::new(0.9, 0., 0.), eye_v, normal_v, 0.),
            Color::white()
        );
        assert_approx_eq_low_prec!(
            light.color_of_illuminated_point(&obj, Point::new(1.1, 0., 0.), eye_v, normal_v, 0.),
            Color::black()
        );
    }
}
