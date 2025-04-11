use crate::render::intersection::IntersecComputations;

pub mod point_light;

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
    use crate::{approx_eq::ApproxEq, Material, Object, Shape};
    use std::f64::consts::FRAC_1_SQRT_2;

    use crate::{
        assert_approx_eq_low_prec,
        light::schlick_reflectance,
        math::{matrix::Matrix, point::Point, tuple::Tuple, vector::Vector},
        render::{intersection::IntersectionCollection, ray::Ray},
    };

    #[test]
    fn schlick_reflectance_under_total_internal_reflection() {
        let sphere = Object::core(Shape::Sphere, Material::glass(), Matrix::identity());

        let ray = Ray::new(Point::new(0., 0., FRAC_1_SQRT_2), Vector::new(0., 1., 0.));
        let intersections = IntersectionCollection::from_times_and_obj(
            ray,
            vec![-FRAC_1_SQRT_2, FRAC_1_SQRT_2],
            &sphere,
        );
        let comps = intersections.hit_computations().unwrap();

        assert_approx_eq_low_prec!(schlick_reflectance(&comps), 1.);
    }

    #[test]
    fn schlick_reflectance_with_perpendicular_viewing_angle() {
        let sphere = Object::core(Shape::Sphere, Material::glass(), Matrix::identity());

        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 1., 0.));
        let intersections = IntersectionCollection::from_times_and_obj(ray, vec![-1., 1.], &sphere);
        let comps = intersections.hit_computations().unwrap();

        assert_approx_eq_low_prec!(schlick_reflectance(&comps), 0.04);
    }

    #[test]
    fn schlick_reflectance_with_small_angle_and_n1_greater_than_n2() {
        let sphere = Object::core(Shape::Sphere, Material::glass(), Matrix::identity());

        let ray = Ray::new(Point::new(0., 0.99, -2.), Vector::new(0., 0., 1.));
        let intersections = IntersectionCollection::from_times_and_obj(ray, vec![1.8589], &sphere);
        let comps = intersections.hit_computations().unwrap();

        assert_approx_eq_low_prec!(schlick_reflectance(&comps), 0.48873);
    }
}
