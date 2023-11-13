use crate::{approx_eq::ApproxEq, transformation::Transform};
use std::ops;

use super::{matrix3::Matrix3, tuple::Tuple};

#[derive(Debug, Clone, Copy)]
pub struct Matrix4 {
    data: [f64; 16],
}

impl Transform for Matrix4 {
    fn get_transformed(self) -> Self {
        self
    }

    fn transform_borrowed(&mut self, transformation_matrix: &Matrix4) {
        *self = (*transformation_matrix) * (*self);
    }
}

impl Matrix4 {
    pub fn new(data: [f64; 16]) -> Self {
        Self { data }
    }
    pub fn empty() -> Self {
        Self::new([0.; 16])
    }

    #[rustfmt::skip]
    pub fn identity_matrix() -> Self {
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
    pub fn submatrix(&self, row_to_del: usize, col_to_del: usize) -> Matrix3 {
        let mut new_data = [0.; 9];
        let mut id = 0;

        for row in 0..4 {
            if row == row_to_del {
                continue;
            }
            for col in 0..4 {
                if col == col_to_del {
                    continue;
                }
                new_data[id] = self.data[row * 4 + col];
                id += 1;
            }
        }
        Matrix3::new(new_data)
    }
    // subject to optimisation
    pub fn minor(&self, row: usize, col: usize) -> f64 {
        self.submatrix(row, col).determinant()
    }
    pub fn cofactor(&self, row: usize, col: usize) -> f64 {
        let minor = self.minor(row, col);
        if (row + col) % 2 == 1 {
            -minor
        } else {
            minor
        }
    }
    pub fn determinant(&self) -> f64 {
        self.data
            .iter()
            .take(4)
            .enumerate()
            .map(|(i, x)| x * self.cofactor(0, i))
            .sum()
    }
    pub fn inverse(&self) -> Option<Matrix4> {
        let determinant = self.determinant();
        if determinant == 0. {
            return None;
        }
        let mut res = Matrix4::empty();

        for row in 0..4 {
            for col in 0..4 {
                let cofactor = self.cofactor(row, col);

                // transpose the matrix here
                res[(col, row)] = cofactor / determinant;
            }
        }
        Some(res)
    }
}

impl PartialEq for Matrix4 {
    fn eq(&self, other: &Matrix4) -> bool {
        self.data
            .iter()
            .enumerate()
            .all(|(id, x)| x.approq_eq(other.data[id]))
    }
}

impl ops::Index<(usize, usize)> for Matrix4 {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        debug_assert!(row < 4);
        debug_assert!(col < 4);
        &self.data[row * 4 + col]
    }
}

impl ops::IndexMut<(usize, usize)> for Matrix4 {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        debug_assert!(row < 4);
        debug_assert!(col < 4);
        &mut self.data[row * 4 + col]
    }
}

impl ops::Mul<Matrix4> for Matrix4 {
    type Output = Self;
    fn mul(self, rhs: Matrix4) -> Self::Output {
        let mut output = Self::empty();
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

impl<T> ops::Mul<T> for Matrix4
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

#[cfg(test)]
mod tests {
    use crate::primitive::point::Point;

    use super::*;

    #[test]
    #[rustfmt::skip]
    fn create_and_index() {
        let matrix = Matrix4::new([
            1.0, 2.0, 3.0, 4.0,
            5.5, 6.5, 7.5, 8.5,
            9.0, 10.0, 11.0, 12.0,
            13.5, 14.5, 15.5, 16.5,
        ]);

        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(0, 3)], 4.0);
        assert_eq!(matrix[(1, 0)], 5.5);
        assert_eq!(matrix[(1, 2)], 7.5);
        assert_eq!(matrix[(2, 2)], 11.0);
        assert_eq!(matrix[(3, 0)], 13.5);
        assert_eq!(matrix[(3, 2)], 15.5);
    }
    #[test]
    fn equality() {
        #[rustfmt::skip]
        let m1 = Matrix4::new([
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);
        #[rustfmt::skip]
        let m2 = Matrix4::new([
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);
        #[rustfmt::skip]
        let other = Matrix4::new([
            0., 2., 3., 4.,
            5., 6., 7., 8.,
            1., 2., 3., 4.,
            5., 6., 7., 8.,
        ]);

        assert_eq!(m1, m2);
        assert_ne!(m1, other);
    }
    #[test]
    fn mul() {
        #[rustfmt::skip]
        let m1 = Matrix4::new([
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);
        #[rustfmt::skip]
        let m2 = Matrix4::new([
            -2., 1., 2., 3.,
            3., 2., 1., -1.,
            4., 3., 6., 5.,
            1., 2., 7., 8.,
        ]);
        #[rustfmt::skip]
        let expected = Matrix4::new([
            20., 22., 50., 48.,
            44., 54., 114., 108.,
            40., 58., 110., 102.,
            16., 26., 46., 42.,
        ]);

        assert_eq!(m1 * m2, expected);
    }
    #[test]
    fn mul_with_tuple() {
        #[rustfmt::skip]
        let m = Matrix4::new([
            1., 2., 3., 4.,
            2., 4., 4., 2.,
            8., 6., 4., 1.,
            0., 0., 0., 1.,
        ]);

        let p = Point::new(1., 2., 3.);
        assert_eq!(m * p, Point::new(18., 24., 33.));
    }
    #[test]
    fn identity_matrix() {
        #[rustfmt::skip]
        let m = Matrix4::new([
            0., 1., 2., 4.,
            1., 2., 4., 8.,
            2., 4., 8., 16.,
            4., 8., 16., 32.,
        ]);

        assert_eq!(m * Matrix4::identity_matrix(), m);
        assert_eq!(Matrix4::identity_matrix() * m, m);
    }
    #[test]
    fn transose() {
        #[rustfmt::skip]
        let m = Matrix4::new([
            0., 9., 3., 0.,
            9., 8., 0., 8.,
            1., 8., 5., 3.,
            0., 0., 5., 8.,
        ]);
        #[rustfmt::skip]
        let expected = Matrix4::new([
            0., 9., 1., 0.,
            9., 8., 8., 0.,
            3., 0., 5., 5.,
            0., 8., 3., 8.,
        ]);
        assert_eq!(m.transpose(), expected);
        assert_eq!(
            Matrix4::identity_matrix().transpose(),
            Matrix4::identity_matrix()
        );
    }

    #[test]
    fn submatrix() {
        #[rustfmt::skip]
        let m = Matrix4::new([
            -6., 1., 1., 6.,
            -8., 5., 8., 6.,
            -1., 0., 8., 2.,
            -7., 1., -1., 1.,
        ]);
        #[rustfmt::skip]
        let expected = Matrix3::new([
            -6., 1., 6.,
            -8., 8., 6.,
            -7., -1., 1.,
        ]);

        assert_eq!(m.submatrix(2, 1), expected);
    }
    #[test]
    fn determinant() {
        #[rustfmt::skip]
        let m = Matrix4::new([
            -2., -8., 3., 5.,
            -3., 1., 7., 3.,
            1., 2., -9., 6.,
            -6., 7., 7., -9.,
        ]);

        assert_eq!(m.cofactor(0, 0), 690.);
        assert_eq!(m.cofactor(0, 1), 447.);
        assert_eq!(m.cofactor(0, 2), 210.);
        assert_eq!(m.cofactor(0, 3), 51.);

        assert_eq!(m.determinant(), -4071.);
    }
    #[test]
    fn inverse() {
        #[rustfmt::skip]
        let m1 = Matrix4::new([
            8., -5., 9., 2.,
            7., 5., 6., 1.,
            -6., 0., 9., 6.,
            -3., 0., -9., -4.,
        ]);
        #[rustfmt::skip]
        let m2 = Matrix4::new([
            9., 3., 0., 9.,
            -5., -2., -6., -3.,
            -4., 9., 6., 4.,
            -7., 6., 6., 2.,
        ]);
        #[rustfmt::skip]
        let i1 = Matrix4::new([
             -0.15385 , -0.15385 , -0.28205 , -0.53846 ,
            -0.07692 , 0.12308 , 0.02564 , 0.03077 ,
            0.35897 , 0.35897 , 0.43590 , 0.92308 ,
            -0.69231 , -0.69231 , -0.76923 , -1.92308 ,
        ]);

        #[rustfmt::skip]
        let i2 = Matrix4::new([
            -0.04074 , -0.07778 , 0.14444 , -0.22222 ,
            -0.07778 , 0.03333 , 0.36667 , -0.33333 ,
            -0.02901 , -0.14630 , -0.10926 , 0.12963 ,
            0.17778 , 0.06667 , -0.26667 , 0.33333 ,
        ]);

        assert_eq!(m1.inverse(), Some(i1));
        assert_eq!(m2.inverse(), Some(i2));
    }
    #[test]
    fn inverse_mul() {
        #[rustfmt::skip]
        let a = Matrix4::new([
            3., -9., 7., 3.,
            3., -8., 2., -9.,
            -4., 4., 4., 1.,
            -6., 4., -1., 1.,
        ]);
        #[rustfmt::skip]
        let b = Matrix4::new([
            8., 2., 2., 2.,
            3., -1., 7., -0.,
            7., 0., 5., 4.,
            6., -2., 0., 5.,
        ]);

        let b_inverse = b.inverse().unwrap();
        assert_eq!(a * b * b_inverse, a);
        assert_eq!(b * b_inverse, Matrix4::identity_matrix());
    }
}
