use crate::primitive::matrix4::Matrix4;

pub trait Transform {
    fn get_transformed(self) -> Self;
    fn transform_borrowed(&mut self, transformation: &Matrix4);

    fn transform(&mut self, transformation: Matrix4) -> &mut Self {
        self.transform_borrowed(&transformation);
        self
    }

    fn translate(&mut self, x: f64, y: f64, z: f64) -> &mut Self {
        self.transform(translation_matrix(x, y, z))
    }
    fn scale(&mut self, x: f64, y: f64, z: f64) -> &mut Self {
        self.transform(scaling_matrix(x, y, z))
    }

    fn rotate_x(&mut self, radians: f64) -> &mut Self {
        self.transform(rotation_x_matrix(radians))
    }
    fn rotate_y(&mut self, radians: f64) -> &mut Self {
        self.transform(rotation_y_matrix(radians))
    }
    fn rotate_z(&mut self, radians: f64) -> &mut Self {
        self.transform(rotation_z_matrix(radians))
    }

    fn sheare(
        &mut self,
        x_prop_y: f64,
        x_prop_z: f64,
        y_prop_x: f64,
        y_prop_z: f64,
        z_prop_x: f64,
        z_prop_y: f64,
    ) -> &mut Self {
        self.transform(shearing_matrix(
            x_prop_y, x_prop_z, y_prop_x, y_prop_z, z_prop_x, z_prop_y,
        ))
    }
}
pub fn translation_matrix(x: f64, y: f64, z: f64) -> Matrix4 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    Matrix4::new([
        1., 0., 0., x,
        0., 1., 0., y,
        0., 0., 1., z,
        0., 0., 0., 1.,
    ])
}
pub fn scaling_matrix(x: f64, y: f64, z: f64) -> Matrix4 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix4::new([
            x, 0., 0., 0.,
            0., y, 0., 0.,
            0., 0., z, 0.,
            0., 0., 0., 1.,
        ])
}
pub fn rotation_x_matrix(radians: f64) -> Matrix4 {
    let sin_r = radians.sin();
    let cos_r = radians.cos();
    #[cfg_attr(rustfmt, rustfmt_skip)]
    Matrix4::new([
        1., 0., 0., 0.,
        0., cos_r, -sin_r, 0.,
        0., sin_r, cos_r, 0.,
        0., 0., 0., 1.,
    ])
}
pub fn rotation_y_matrix(radians: f64) -> Matrix4 {
    let sin_r = radians.sin();
    let cos_r = radians.cos();
    #[cfg_attr(rustfmt, rustfmt_skip)]
    Matrix4::new([
        cos_r, 0., sin_r, 0.,
        0., 1., 0., 0.,
        -sin_r, 0., cos_r, 0.,
        0., 0., 0., 1.,
    ])
}
pub fn rotation_z_matrix(radians: f64) -> Matrix4 {
    let sin_r = radians.sin();
    let cos_r = radians.cos();
    #[cfg_attr(rustfmt, rustfmt_skip)]
    Matrix4::new([
        cos_r, -sin_r, 0., 0.,
        sin_r, cos_r, 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.,
    ])
}

pub fn shearing_matrix(
    x_prop_y: f64,
    x_prop_z: f64,
    y_prop_x: f64,
    y_prop_z: f64,
    z_prop_x: f64,
    z_prop_y: f64,
) -> Matrix4 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    Matrix4::new([
        1., x_prop_y, x_prop_z, 0.,
        y_prop_x, 1., y_prop_z, 0.,
        z_prop_x, z_prop_y, 1., 0.,
        0., 0., 0., 1.,
    ])
}

#[cfg(test)]
mod tests {
    use std::f64::consts;

    use crate::primitive::{point::Point, tuple::Tuple, vector::Vector};

    use super::*;

    #[test]
    fn translate_point() {
        assert_eq!(
            translation_matrix(5., -3., 2.) * Point::new(-3., 4., 5.),
            Point::new(2., 1., 7.)
        );
    }

    #[test]
    fn inverse_translate_point() {
        assert_eq!(
            translation_matrix(5., -3., 2.).inverse().unwrap() * Point::new(-3., 4., 5.),
            Point::new(-8., 7., 3.)
        );
    }
    #[test]
    fn translate_vector() {
        let v = Vector::new(-3., 4., 5.);
        assert_eq!(translation_matrix(5., -3., 2.) * v, v);
    }

    #[test]
    fn scale_point() {
        assert_eq!(
            scaling_matrix(2., 3., 4.) * Point::new(-4., 6., 8.),
            Point::new(-8., 18., 32.)
        );
    }
    #[test]
    fn scale_vector() {
        assert_eq!(
            scaling_matrix(2., 3., 4.) * Vector::new(-4., 6., 8.),
            Vector::new(-8., 18., 32.)
        );
    }
    #[test]
    fn inverse_scale_vector() {
        assert_eq!(
            scaling_matrix(2., 3., 4.).inverse().unwrap() * Vector::new(-4., 6., 8.),
            Vector::new(-2., 2., 2.)
        );
    }
    #[test]
    fn reflect_by_scale() {
        assert_eq!(
            scaling_matrix(-1., 1., 1.) * Point::new(2., 3., 4.),
            Point::new(-2., 3., 4.)
        );
    }
    #[test]
    fn rotate_around_x() {
        let half_quarter = rotation_x_matrix(consts::FRAC_PI_4);
        let full_quarter = rotation_x_matrix(consts::FRAC_PI_2);
        let p = Point::new(0., 1., 0.);

        assert_eq!(
            half_quarter * p,
            Point::new(0., consts::SQRT_2 / 2., consts::SQRT_2 / 2.)
        );

        assert_eq!(full_quarter * p, Point::new(0., 0., 1.));
    }
    #[test]
    fn inverse_rotate_around_x() {
        let half_quarter = rotation_x_matrix(consts::FRAC_PI_4);
        let p = Point::new(0., 1., 0.);

        assert_eq!(
            half_quarter.inverse(),
            Some(rotation_x_matrix(-consts::FRAC_PI_4))
        );
        assert_eq!(
            half_quarter.inverse().unwrap() * p,
            Point::new(0., consts::SQRT_2 / 2., -consts::SQRT_2 / 2.)
        );
    }
    #[test]
    fn rotate_around_y() {
        let half_quarter = rotation_y_matrix(consts::FRAC_PI_4);
        let full_quarter = rotation_y_matrix(consts::FRAC_PI_2);
        let p = Point::new(0., 0., 1.);

        assert_eq!(
            half_quarter * p,
            Point::new(consts::SQRT_2 / 2., 0., consts::SQRT_2 / 2.)
        );

        assert_eq!(full_quarter * p, Point::new(1., 0., 0.));
    }
    #[test]
    fn rotate_around_z() {
        let half_quarter = rotation_z_matrix(consts::FRAC_PI_4);
        let full_quarter = rotation_z_matrix(consts::FRAC_PI_2);
        let p = Point::new(0., 1., 0.);

        assert_eq!(
            half_quarter * p,
            Point::new(-consts::SQRT_2 / 2., consts::SQRT_2 / 2., 0.)
        );

        assert_eq!(full_quarter * p, Point::new(-1., 0., 0.));
    }

    #[test]
    fn sheare() {
        let p = Point::new(2., 3., 4.);

        assert_eq!(
            shearing_matrix(1., 0., 0., 0., 0., 0.) * p,
            Point::new(5., 3., 4.)
        );
        assert_eq!(
            shearing_matrix(0., 1., 0., 0., 0., 0.) * p,
            Point::new(6., 3., 4.)
        );
        assert_eq!(
            shearing_matrix(0., 0., 1., 0., 0., 0.) * p,
            Point::new(2., 5., 4.)
        );
        assert_eq!(
            shearing_matrix(0., 0., 0., 1., 0., 0.) * p,
            Point::new(2., 7., 4.)
        );
        assert_eq!(
            shearing_matrix(0., 0., 0., 0., 1., 0.) * p,
            Point::new(2., 3., 6.)
        );
        assert_eq!(
            shearing_matrix(0., 0., 0., 0., 0., 1.) * p,
            Point::new(2., 3., 7.)
        );
    }

    #[test]
    fn transform_matrix() {
        assert_eq!(
            Matrix4::identiy_matrix()
                .scale(1., 0., -1.,)
                .translate(2., 10., -0.5)
                .get_transformed(),
            translation_matrix(2., 10., -0.5) * scaling_matrix(1., 0., -1.)
        );
    }
}
