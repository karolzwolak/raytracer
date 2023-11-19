use crate::primitive::{matrix4::Matrix4, point::Point, tuple::Tuple, vector::Vector};

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

#[rustfmt::skip]
pub fn translation_matrix(x: f64, y: f64, z: f64) -> Matrix4 {
     Matrix4::new([
        1., 0., 0., x,
        0., 1., 0., y,
        0., 0., 1., z,
        0., 0., 0., 1.,
    ])
}

#[rustfmt::skip]
pub fn scaling_matrix(x: f64, y: f64, z: f64) -> Matrix4 {
        Matrix4::new([
            x, 0., 0., 0.,
            0., y, 0., 0.,
            0., 0., z, 0.,
            0., 0., 0., 1.,
        ])
}

#[rustfmt::skip]
pub fn rotation_x_matrix(radians: f64) -> Matrix4 {
    let sin_r = radians.sin();
    let cos_r = radians.cos();
    Matrix4::new([
        1., 0., 0., 0.,
        0., cos_r, -sin_r, 0.,
        0., sin_r, cos_r, 0.,
        0., 0., 0., 1.,
    ])
}

#[rustfmt::skip]
pub fn rotation_y_matrix(radians: f64) -> Matrix4 {
    let sin_r = radians.sin();
    let cos_r = radians.cos();
    Matrix4::new([
        cos_r, 0., sin_r, 0.,
        0., 1., 0., 0.,
        -sin_r, 0., cos_r, 0.,
        0., 0., 0., 1.,
    ])
}

#[rustfmt::skip]
pub fn rotation_z_matrix(radians: f64) -> Matrix4 {
    let sin_r = radians.sin();
    let cos_r = radians.cos();
    Matrix4::new([
        cos_r, -sin_r, 0., 0.,
        sin_r, cos_r, 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.,
    ])
}

#[rustfmt::skip]
pub fn shearing_matrix(
    x_prop_y: f64,
    x_prop_z: f64,
    y_prop_x: f64,
    y_prop_z: f64,
    z_prop_x: f64,
    z_prop_y: f64,
) -> Matrix4 {
    Matrix4::new([
        1., x_prop_y, x_prop_z, 0.,
        y_prop_x, 1., y_prop_z, 0.,
        z_prop_x, z_prop_y, 1., 0.,
        0., 0., 0., 1.,
    ])
}

pub fn view_tranformation_matrix(from: Point, to: Point, up_v: Vector) -> Matrix4 {
    let up_v = up_v.normalize();

    let forward_v = (to - from).normalize();
    let left_v = forward_v.cross(up_v);
    let true_up_v = left_v.cross(forward_v);

    #[rustfmt::skip]
    let orientation = Matrix4::new([
        left_v.x(), left_v.y(), left_v.z(), 0.,
        true_up_v.x(), true_up_v.y(), true_up_v.z(), 0.,
        -forward_v.x(), -forward_v.y(), -forward_v.z(), 0.,
        0., 0., 0., 1.,
    ]);

    orientation * translation_matrix(-from.x(), -from.y(), -from.z())
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
            Matrix4::identity_matrix()
                .scale(1., 0., -1.,)
                .translate(2., 10., -0.5)
                .get_transformed(),
            translation_matrix(2., 10., -0.5) * scaling_matrix(1., 0., -1.)
        );
    }

    #[test]
    fn view_default_transformation() {
        let from = Point::zero();
        let to = Point::new(0., 0., -1.);
        let up_v = Vector::new(0., 1., 0.);

        assert_eq!(
            view_tranformation_matrix(from, to, up_v),
            Matrix4::identity_matrix()
        )
    }
    #[test]
    fn view_transformation_looking_in_positive_z_dir() {
        let from = Point::zero();
        let to = Point::new(0., 0., 1.);
        let up_v = Vector::new(0., 1., 0.);

        assert_eq!(
            view_tranformation_matrix(from, to, up_v),
            scaling_matrix(-1., 1., -1.)
        )
    }
    #[test]
    fn view_transormation_moves_the_world() {
        let from = Point::new(0., 0., 8.);
        let to = Point::zero();
        let up_v = Vector::new(0., 1., 0.);

        assert_eq!(
            view_tranformation_matrix(from, to, up_v),
            translation_matrix(0., 0., -8.)
        )
    }
    #[test]
    fn arbitrary_view_transformation() {
        let from = Point::new(1., 3., 2.);
        let to = Point::new(4., -2., 8.);
        let up_v = Vector::new(1., 1., 0.);

        #[rustfmt::skip]
        let expected = Matrix4::new([
            -0.50709, 0.50709, 0.67612, -2.36643,
            0.76772, 0.60609, 0.12122, -2.82843,
            -0.35857,0.59761, -0.71714, 0.00000,
            0.00000, 0.00000, 0.00000, 1.00000,
        ]);

        assert_eq!(view_tranformation_matrix(from, to, up_v), expected);
    }
}
