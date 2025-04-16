use std::ops;

use super::{point::Point, transform::Transform, tuple::Tuple, vector::Vector};
use crate::math::approx_eq::{self, ApproxEq};

#[derive(Debug, Clone, Copy)]
/// Simple 4x4 matrix
pub struct Matrix {
    data: [f64; 16],
}

impl Transform for Matrix {
    fn transform(&mut self, matrix: &Matrix) {
        *self = self.transform_new(matrix);
    }

    fn transform_new(&self, matrix: &Matrix) -> Self {
        matrix * (self as &Matrix)
    }
}

impl From<&[Matrix]> for Matrix {
    fn from(val: &[Matrix]) -> Self {
        Matrix::from_iter(val.iter().copied())
    }
}

impl<A> FromIterator<A> for Matrix
where
    Matrix: From<A>,
{
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        iter.into_iter().fold(Matrix::identity(), |acc, m| {
            acc.transform_new(&Matrix::from(m))
        })
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Self::identity()
    }
}

impl Matrix {
    pub fn new(data: [f64; 16]) -> Self {
        Self { data }
    }
    pub fn empty() -> Self {
        Self::new([0.; 16])
    }

    #[rustfmt::skip]
    pub fn identity() -> Self {
        Self::new([
            1., 0., 0., 0.,
            0., 1., 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        ])
    }

    pub fn transpose(&self) -> Self {
        let mut res = *self;

        res.data.swap(1, 4);
        res.data.swap(2, 8);

        res.data.swap(3, 12);
        res.data.swap(6, 9);

        res.data.swap(7, 13);
        res.data.swap(11, 14);

        res
    }
    pub fn mul_transposed<T: Tuple>(&self, rhs: T) -> T {
        T::new(
            self[(0, 0)] * rhs.x()
                + self[(1, 0)] * rhs.y()
                + self[(2, 0)] * rhs.z()
                + self[(3, 0)] * rhs.w(),
            self[(0, 1)] * rhs.x()
                + self[(1, 1)] * rhs.y()
                + self[(2, 1)] * rhs.z()
                + self[(3, 1)] * rhs.w(),
            self[(0, 2)] * rhs.x()
                + self[(1, 2)] * rhs.y()
                + self[(2, 2)] * rhs.z()
                + self[(3, 2)] * rhs.w(),
        )
    }
    pub fn inverse(&self) -> Option<Matrix> {
        let mut res = Matrix::identity();

        let mut copy = *self;
        let mut row_of_one_for_col = [0, 1, 2, 3];
        for i in 0..4 {
            for row in i..4 {
                if !copy[(row, i)].approx_eq(&0.) {
                    row_of_one_for_col[i] = row;
                    break;
                }
            }

            let actual_row = row_of_one_for_col[i];
            if i != actual_row {
                for col in 0..4 {
                    let old = copy[(actual_row, col)];
                    copy[(actual_row, col)] = copy[(i, col)];
                    copy[(i, col)] = old;

                    let old = res[(actual_row, col)];
                    res[(actual_row, col)] = res[(i, col)];
                    res[(i, col)] = old;
                }
            }
            let row = i;

            let factor_to_1 = copy[(row, i)];
            if factor_to_1.approx_eq(&0.) {
                return None;
            }

            for col in 0..4 {
                copy[(row, col)] /= factor_to_1;
                res[(row, col)] /= factor_to_1;
            }

            for inner_row in 0..4 {
                let row_factor = copy[(inner_row, i)];

                for inner_col in 0..4 {
                    if row == inner_row {
                        continue;
                    }
                    copy[(inner_row, inner_col)] -= copy[(i, inner_col)] * row_factor;
                    res[(inner_row, inner_col)] -= res[(i, inner_col)] * row_factor;
                }
            }
        }

        Some(res)
    }


    #[rustfmt::skip]
    pub fn translation(x: f64, y: f64, z: f64) -> Matrix {
        Matrix::new([
            1., 0., 0., x,
            0., 1., 0., y,
            0., 0., 1., z,
            0., 0., 0., 1.,
        ])
    }

    #[rustfmt::skip]
    pub fn scaling(x: f64, y: f64, z: f64) -> Matrix {
        let x = if x.approx_eq(&0.) { approx_eq::EPSILON } else { x };
        let y = if y.approx_eq(&0.) { approx_eq::EPSILON } else { y };
        let z = if z.approx_eq(&0.) { approx_eq::EPSILON } else { z };

        Matrix::new([
            x, 0., 0., 0.,
            0., y, 0., 0.,
            0., 0., z, 0.,
            0., 0., 0., 1.,
        ])
    }

    #[rustfmt::skip]
    pub fn scaling_uniform(f: f64) -> Matrix {
        Self::scaling(f, f, f)
    }

    #[rustfmt::skip]
    pub fn rotation_x(radians: f64) -> Matrix {
        let sin_r = radians.sin();
        let cos_r = radians.cos();
        Matrix::new([
            1., 0., 0., 0.,
            0., cos_r, -sin_r, 0.,
            0., sin_r, cos_r, 0.,
            0., 0., 0., 1.,
        ])
    }

    #[rustfmt::skip]
    pub fn rotation_y(radians: f64) -> Matrix {
        let sin_r = radians.sin();
        let cos_r = radians.cos();
        Matrix::new([
            cos_r, 0., sin_r, 0.,
            0., 1., 0., 0.,
            -sin_r, 0., cos_r, 0.,
            0., 0., 0., 1.,
        ])
    }

    #[rustfmt::skip]
    pub fn rotation_z(radians: f64) -> Matrix {
        let sin_r = radians.sin();
        let cos_r = radians.cos();
        Matrix::new([
            cos_r, -sin_r, 0., 0.,
            sin_r, cos_r, 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        ])
    }

    #[rustfmt::skip]
    pub fn shearing(
        x_prop_y: f64,
        x_prop_z: f64,
        y_prop_x: f64,
        y_prop_z: f64,
        z_prop_x: f64,
        z_prop_y: f64,
    ) -> Matrix {
        Matrix::new([
            1., x_prop_y, x_prop_z, 0.,
            y_prop_x, 1., y_prop_z, 0.,
            z_prop_x, z_prop_y, 1., 0.,
            0., 0., 0., 1.,
        ])
    }

    pub fn view_tranformation(from: Point, to: Point, up_v: Vector) -> Matrix {
        let up_v = up_v.normalize();

        let forward_v = (to - from).normalize();
        let left_v = forward_v.cross(up_v);
        let true_up_v = left_v.cross(forward_v);

        #[rustfmt::skip]
    let orientation = Matrix::new([
        left_v.x(), left_v.y(), left_v.z(), 0.,
        true_up_v.x(), true_up_v.y(), true_up_v.z(), 0.,
        -forward_v.x(), -forward_v.y(), -forward_v.z(), 0.,
        0., 0., 0., 1.,
    ]);

        orientation * Matrix::translation(-from.x(), -from.y(), -from.z())
    }
}

impl ApproxEq for Matrix {
    fn approx_eq_epsilon(&self, other: &Self, epsilon: f64) -> bool {
        self.data
            .iter()
            .enumerate()
            .all(|(id, x)| x.approx_eq_epsilon(&other.data[id], epsilon))
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Matrix) -> bool {
        self.approx_eq(other)
    }
}

impl ops::Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        debug_assert!(row < 4);
        debug_assert!(col < 4);
        &self.data[row * 4 + col]
    }
}

impl ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        debug_assert!(row < 4);
        debug_assert!(col < 4);
        &mut self.data[row * 4 + col]
    }
}

impl ops::Mul<Matrix> for Matrix {
    type Output = Self;
    fn mul(self, rhs: Matrix) -> Self::Output {
        &self * &rhs
    }
}

impl ops::Mul<&Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        let mut output = Self::Output::empty();
        for row in 0..4 {
            for col in 0..4 {
                output[(row, col)] = self[(row, 0)] * rhs[(0, col)]
                    + self[(row, 1)] * rhs[(1, col)]
                    + self[(row, 2)] * rhs[(2, col)]
                    + self[(row, 3)] * rhs[(3, col)];
            }
        }
        output
    }
}

impl ops::MulAssign<Matrix> for Matrix {
    fn mul_assign(&mut self, rhs: Self) {
        *self *= &rhs;
    }
}

impl ops::MulAssign<&Matrix> for Matrix {
    fn mul_assign(&mut self, rhs: &Matrix) {
        *self = (self as &Matrix) * rhs;
    }
}

impl<T> ops::Mul<T> for &Matrix
where
    T: Tuple,
{
    type Output = T;
    fn mul(self, rhs: T) -> Self::Output {
        T::new(
            self[(0, 0)] * rhs.x()
                + self[(0, 1)] * rhs.y()
                + self[(0, 2)] * rhs.z()
                + self[(0, 3)] * rhs.w(),
            self[(1, 0)] * rhs.x()
                + self[(1, 1)] * rhs.y()
                + self[(1, 2)] * rhs.z()
                + self[(1, 3)] * rhs.w(),
            self[(2, 0)] * rhs.x()
                + self[(2, 1)] * rhs.y()
                + self[(2, 2)] * rhs.z()
                + self[(2, 3)] * rhs.w(),
        )
    }
}
impl<T> ops::Mul<T> for Matrix
where
    T: Tuple,
{
    type Output = T;
    #[allow(clippy::op_ref)]
    fn mul(self, rhs: T) -> Self::Output {
        &self * rhs
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts;

    use super::*;
    use crate::{
        assert_approx_eq_low_prec,
        math::{point::Point, vector::Vector},
    };

    #[test]
    #[rustfmt::skip]
    fn create_and_index() {
        let matrix = Matrix::new([
            1.0, 2.0, 3.0, 4.0,
            5.5, 6.5, 7.5, 8.5,
            9.0, 10.0, 11.0, 12.0,
            13.5, 14.5, 15.5, 16.5,
        ]);

        assert_approx_eq_low_prec!(matrix[(0, 0)], 1.0);
        assert_approx_eq_low_prec!(matrix[(0, 3)], 4.0);
        assert_approx_eq_low_prec!(matrix[(1, 0)], 5.5);
        assert_approx_eq_low_prec!(matrix[(1, 2)], 7.5);
        assert_approx_eq_low_prec!(matrix[(2, 2)], 11.0);
        assert_approx_eq_low_prec!(matrix[(3, 0)], 13.5);
        assert_approx_eq_low_prec!(matrix[(3, 2)], 15.5);
    }
    #[test]
    fn equality() {
        #[rustfmt::skip]
        let m1 = Matrix::new([
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);
        #[rustfmt::skip]
        let m2 = Matrix::new([
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);
        #[rustfmt::skip]
        let other = Matrix::new([
            0., 2., 3., 4.,
            5., 6., 7., 8.,
            1., 2., 3., 4.,
            5., 6., 7., 8.,
        ]);

        assert_approx_eq_low_prec!(m1, m2);
        assert_ne!(m1, other);
    }
    #[test]
    fn mul() {
        #[rustfmt::skip]
        let m1 = Matrix::new([
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);
        #[rustfmt::skip]
        let m2 = Matrix::new([
            -2., 1., 2., 3.,
            3., 2., 1., -1.,
            4., 3., 6., 5.,
            1., 2., 7., 8.,
        ]);
        #[rustfmt::skip]
        let expected = Matrix::new([
            20., 22., 50., 48.,
            44., 54., 114., 108.,
            40., 58., 110., 102.,
            16., 26., 46., 42.,
        ]);

        assert_approx_eq_low_prec!(m1 * m2, expected);
    }
    #[test]
    fn mul_assign() {
        #[rustfmt::skip]
        let mut m1 = Matrix::new([
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);
        #[rustfmt::skip]
        let m2 = Matrix::new([
            -2., 1., 2., 3.,
            3., 2., 1., -1.,
            4., 3., 6., 5.,
            1., 2., 7., 8.,
        ]);
        #[rustfmt::skip]
        let expected = Matrix::new([
            20., 22., 50., 48.,
            44., 54., 114., 108.,
            40., 58., 110., 102.,
            16., 26., 46., 42.,
        ]);

        m1 *= m2;
        assert_approx_eq_low_prec!(m1, expected);
    }
    #[test]
    fn mul_with_tuple() {
        #[rustfmt::skip]
        let m = Matrix::new([
            1., 2., 3., 4.,
            2., 4., 4., 2.,
            8., 6., 4., 1.,
            0., 0., 0., 1.,
        ]);

        let p = Point::new(1., 2., 3.);
        assert_approx_eq_low_prec!(m * p, Point::new(18., 24., 33.));
    }
    #[test]
    fn identity_matrix() {
        #[rustfmt::skip]
        let m = Matrix::new([
            0., 1., 2., 4.,
            1., 2., 4., 8.,
            2., 4., 8., 16.,
            4., 8., 16., 32.,
        ]);

        assert_approx_eq_low_prec!(m * Matrix::identity(), m);
        assert_approx_eq_low_prec!(Matrix::identity() * m, m);
    }
    #[test]
    fn transpose() {
        #[rustfmt::skip]
        let m = Matrix::new([
            0., 9., 3., 0.,
            9., 8., 0., 8.,
            1., 8., 5., 3.,
            0., 0., 5., 8.,
        ]);
        #[rustfmt::skip]
        let expected = Matrix::new([
            0., 9., 1., 0.,
            9., 8., 8., 0.,
            3., 0., 5., 5.,
            0., 8., 3., 8.,
        ]);
        assert_approx_eq_low_prec!(m.transpose(), expected);
        assert_approx_eq_low_prec!(Matrix::identity().transpose(), Matrix::identity());
    }
    #[test]
    fn mul_transposed() {
        #[rustfmt::skip]
        let m = Matrix::new([
            0., 9., 3., 0.,
            9., 8., 0., 8.,
            1., 8., 5., 3.,
            0., 0., 5., 8.,
        ]);
        #[rustfmt::skip]
        let transposed = Matrix::new([
            0., 9., 1., 0.,
            9., 8., 8., 0.,
            3., 0., 5., 5.,
            0., 8., 3., 8.,
        ]);

        let v = Vector::new(1., -2., 0.5);
        assert_approx_eq_low_prec!(m.transpose(), transposed);
        assert_approx_eq_low_prec!(m.mul_transposed(v), transposed * v);
    }
    #[test]
    fn inverse() {
        #[rustfmt::skip]
        let m1 = Matrix::new([
            8., -5., 9., 2.,
            7., 5., 6., 1.,
            -6., 0., 9., 6.,
            -3., 0., -9., -4.,
        ]);
        #[rustfmt::skip]
        let m2 = Matrix::new([
            9., 3., 0., 9.,
            -5., -2., -6., -3.,
            -4., 9., 6., 4.,
            -7., 6., 6., 2.,
        ]);
        #[rustfmt::skip]
        let i1 = Matrix::new([
             -0.15385 , -0.15385 , -0.28205 , -0.53846 ,
            -0.07692 , 0.12308 , 0.02564 , 0.03077 ,
            0.35897 , 0.35897 , 0.43590 , 0.92308 ,
            -0.69231 , -0.69231 , -0.76923 , -1.92308 ,
        ]);

        #[rustfmt::skip]
        let i2 = Matrix::new([
            -0.04074 , -0.07778 , 0.14444 , -0.22222 ,
            -0.07778 , 0.03333 , 0.36667 , -0.33333 ,
            -0.02901 , -0.14630 , -0.10926 , 0.12963 ,
            0.17778 , 0.06667 , -0.26667 , 0.33333 ,
        ]);

        assert_approx_eq_low_prec!(m1.inverse().unwrap(), i1);
        assert_approx_eq_low_prec!(m2.inverse().unwrap(), i2);
    }

    #[test]
    fn inverse_zero_on_diagonal() {
        #[rustfmt::skip]
        let m = Matrix::new([
            1., 1., 0., 1.,
            1., 1., 0., 0.,
            1., 0., 0., 1.,
            0., 1., 1., 0.,
        ]);

        #[rustfmt::skip]
        let i = Matrix::new([
            -1., 1., 1., 0.,
            1., 0., -1., 0.,
            -1., 0., 1., 1.,
            1., -1., 0., 0.,
        ]);

        assert_approx_eq_low_prec!(m.inverse().unwrap(), i);
    }

    #[test]
    fn inverse_mul() {
        #[rustfmt::skip]
        let a = Matrix::new([
            3., -9., 7., 3.,
            3., -8., 2., -9.,
            -4., 4., 4., 1.,
            -6., 4., -1., 1.,
        ]);
        #[rustfmt::skip]
        let b = Matrix::new([
            8., 2., 2., 2.,
            3., -1., 7., -0.,
            7., 0., 5., 4.,
            6., -2., 0., 5.,
        ]);

        let b_inverse = b.inverse().unwrap();
        assert_approx_eq_low_prec!(a * b * b_inverse, a);
        assert_approx_eq_low_prec!(b * b_inverse, Matrix::identity());
    }

    #[test]
    fn translate_point() {
        assert_approx_eq_low_prec!(
            Matrix::translation(5., -3., 2.) * Point::new(-3., 4., 5.),
            Point::new(2., 1., 7.)
        );
    }

    #[test]
    fn inverse_translate_point() {
        assert_approx_eq_low_prec!(
            Matrix::translation(5., -3., 2.).inverse().unwrap() * Point::new(-3., 4., 5.),
            Point::new(-8., 7., 3.)
        );
    }
    #[test]
    fn translate_vector() {
        let v = Vector::new(-3., 4., 5.);
        assert_approx_eq_low_prec!(Matrix::translation(5., -3., 2.) * v, v);
    }

    #[test]
    fn scale_point() {
        assert_approx_eq_low_prec!(
            Matrix::scaling(2., 3., 4.) * Point::new(-4., 6., 8.),
            Point::new(-8., 18., 32.)
        );
    }
    #[test]
    fn scale_vector() {
        assert_approx_eq_low_prec!(
            Matrix::scaling(2., 3., 4.) * Vector::new(-4., 6., 8.),
            Vector::new(-8., 18., 32.)
        );
    }
    #[test]
    fn inverse_scale_vector() {
        assert_approx_eq_low_prec!(
            Matrix::scaling(2., 3., 4.).inverse().unwrap() * Vector::new(-4., 6., 8.),
            Vector::new(-2., 2., 2.)
        );
    }
    #[test]
    fn reflect_by_scale() {
        assert_approx_eq_low_prec!(
            Matrix::scaling(-1., 1., 1.) * Point::new(2., 3., 4.),
            Point::new(-2., 3., 4.)
        );
    }
    #[test]
    fn rotate_around_x() {
        let half_quarter = Matrix::rotation_x(consts::FRAC_PI_4);
        let full_quarter = Matrix::rotation_x(consts::FRAC_PI_2);
        let p = Point::new(0., 1., 0.);

        assert_approx_eq_low_prec!(
            half_quarter * p,
            Point::new(0., consts::SQRT_2 / 2., consts::SQRT_2 / 2.)
        );

        assert_approx_eq_low_prec!(full_quarter * p, Point::new(0., 0., 1.));
    }
    #[test]
    fn inverse_rotate_around_x() {
        let half_quarter = Matrix::rotation_x(consts::FRAC_PI_4);
        let p = Point::new(0., 1., 0.);

        assert_approx_eq_low_prec!(
            half_quarter.inverse().unwrap(),
            Matrix::rotation_x(-consts::FRAC_PI_4)
        );
        assert_approx_eq_low_prec!(
            half_quarter.inverse().unwrap() * p,
            Point::new(0., consts::SQRT_2 / 2., -consts::SQRT_2 / 2.)
        );
    }
    #[test]
    fn rotate_around_y() {
        let half_quarter = Matrix::rotation_y(consts::FRAC_PI_4);
        let full_quarter = Matrix::rotation_y(consts::FRAC_PI_2);
        let p = Point::new(0., 0., 1.);

        assert_approx_eq_low_prec!(
            half_quarter * p,
            Point::new(consts::SQRT_2 / 2., 0., consts::SQRT_2 / 2.)
        );

        assert_approx_eq_low_prec!(full_quarter * p, Point::new(1., 0., 0.));
    }
    #[test]
    fn rotate_around_z() {
        let half_quarter = Matrix::rotation_z(consts::FRAC_PI_4);
        let full_quarter = Matrix::rotation_z(consts::FRAC_PI_2);
        let p = Point::new(0., 1., 0.);

        assert_approx_eq_low_prec!(
            half_quarter * p,
            Point::new(-consts::SQRT_2 / 2., consts::SQRT_2 / 2., 0.)
        );

        assert_approx_eq_low_prec!(full_quarter * p, Point::new(-1., 0., 0.));
    }

    #[test]
    fn sheare() {
        let p = Point::new(2., 3., 4.);

        assert_approx_eq_low_prec!(
            Matrix::shearing(1., 0., 0., 0., 0., 0.) * p,
            Point::new(5., 3., 4.)
        );
        assert_approx_eq_low_prec!(
            Matrix::shearing(0., 1., 0., 0., 0., 0.) * p,
            Point::new(6., 3., 4.)
        );
        assert_approx_eq_low_prec!(
            Matrix::shearing(0., 0., 1., 0., 0., 0.) * p,
            Point::new(2., 5., 4.)
        );
        assert_approx_eq_low_prec!(
            Matrix::shearing(0., 0., 0., 1., 0., 0.) * p,
            Point::new(2., 7., 4.)
        );
        assert_approx_eq_low_prec!(
            Matrix::shearing(0., 0., 0., 0., 1., 0.) * p,
            Point::new(2., 3., 6.)
        );
        assert_approx_eq_low_prec!(
            Matrix::shearing(0., 0., 0., 0., 0., 1.) * p,
            Point::new(2., 3., 7.)
        );
    }

    #[test]
    fn transform_matrix() {
        assert_approx_eq_low_prec!(
            Matrix::identity()
                .scale(1., 0., -1.,)
                .translate(2., 10., -0.5)
                .transformed(),
            Matrix::translation(2., 10., -0.5) * Matrix::scaling(1., 0., -1.)
        );
    }

    #[test]
    fn view_default_transformation() {
        let from = Point::zero();
        let to = Point::new(0., 0., -1.);
        let up_v = Vector::new(0., 1., 0.);

        assert_approx_eq_low_prec!(
            Matrix::view_tranformation(from, to, up_v),
            Matrix::identity()
        )
    }
    #[test]
    fn view_transformation_looking_in_positive_z_dir() {
        let from = Point::zero();
        let to = Point::new(0., 0., 1.);
        let up_v = Vector::new(0., 1., 0.);

        assert_approx_eq_low_prec!(
            Matrix::view_tranformation(from, to, up_v),
            Matrix::scaling(-1., 1., -1.)
        )
    }
    #[test]
    fn view_transformation_moves_the_scene() {
        let from = Point::new(0., 0., 8.);
        let to = Point::zero();
        let up_v = Vector::new(0., 1., 0.);

        assert_approx_eq_low_prec!(
            Matrix::view_tranformation(from, to, up_v),
            Matrix::translation(0., 0., -8.)
        )
    }
    #[test]
    fn arbitrary_view_transformation() {
        let from = Point::new(1., 3., 2.);
        let to = Point::new(4., -2., 8.);
        let up_v = Vector::new(1., 1., 0.);

        #[rustfmt::skip]
        let expected = Matrix::new([
            -0.50709, 0.50709, 0.67612, -2.36643,
            0.76772, 0.60609, 0.12122, -2.82843,
            -0.35857,0.59761, -0.71714, 0.00000,
            0.00000, 0.00000, 0.00000, 1.00000,
        ]);

        assert_approx_eq_low_prec!(Matrix::view_tranformation(from, to, up_v), expected);
    }

    #[test]
    fn scaling_uniform_scales_by_same_factor_on_all_dimentions() {
        assert_approx_eq_low_prec!(Matrix::scaling_uniform(2.), Matrix::scaling(2., 2., 2.));
    }

    fn test_scale(factor: f64) {
        let m = Matrix::scaling(factor, 1., 1.);
        let _ = m.inverse().unwrap(); // make sure the matrix is invertible

        let p = Point::new(1., 2., 3.);
        assert_approx_eq_low_prec!(m * p, Point::new(factor, 2., 3.));
    }

    #[test]
    fn scaling_by_inf() {
        test_scale(f64::INFINITY);
    }

    #[test]
    fn scaling_by_zero() {
        test_scale(0.);
    }

    #[test]
    fn scaling_by_nearly_zero() {
        test_scale(approx_eq::EPSILON / 100.);
    }
}
