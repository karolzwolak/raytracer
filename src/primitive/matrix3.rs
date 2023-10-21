use crate::approx_eq::ApproxEq;
use std::ops;

use super::{matrix2::Matrix2, tuple::Tuple};

#[derive(Debug, Clone, Copy)]
pub struct Matrix3 {
    data: [f64; 9],
}

impl Matrix3 {
    pub fn new(data: [f64; 9]) -> Self {
        Self { data }
    }
    pub fn empty() -> Self {
        Self::new([0.; 9])
    }
    pub fn identiy_matrix() -> Self {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Self::new([
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.,
        ])
    }

    pub fn transpose(&self) -> Self {
        let mut res = self.clone();

        res.data.swap(1, 3);
        res.data.swap(2, 6);
        res.data.swap(5, 7);

        res
    }

    pub fn submatrix(&self, row_to_del: usize, col_to_del: usize) -> Matrix2 {
        let mut new_data = [0.; 4];
        let mut id = 0;

        for row in 0..3 {
            if row == row_to_del {
                continue;
            }
            for col in 0..3 {
                if col == col_to_del {
                    continue;
                }
                new_data[id] = self.data[row * 3 + col];
                id += 1;
            }
        }
        Matrix2::new(new_data)
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
            .take(3)
            .enumerate()
            .map(|(i, x)| x * self.cofactor(0, i))
            .sum()
    }
}

impl PartialEq for Matrix3 {
    fn eq(&self, other: &Matrix3) -> bool {
        self.data
            .iter()
            .enumerate()
            .all(|(id, x)| x.approq_eq(other.data[id]))
    }
}

impl ops::Index<(usize, usize)> for Matrix3 {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        debug_assert!(row < 3);
        debug_assert!(col < 3);
        &self.data[row * 3 + col]
    }
}

impl ops::IndexMut<(usize, usize)> for Matrix3 {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        debug_assert!(row < 3);
        debug_assert!(col < 3);
        &mut self.data[row * 3 + col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose() {
        #[cfg_attr(rustfmt,rustfmt_skip)]
        let m = Matrix3::new([
            0., 1., 2.,
            3., 4., 5.,
            6., 7., 8.
        ]);
        #[cfg_attr(rustfmt,rustfmt_skip)]
        let expected = Matrix3::new([
            0., 3., 6.,
            1., 4., 7.,
            2., 5., 8.
        ]);
        assert_eq!(m.transpose(), expected);
    }
    #[test]
    fn submatrix() {
        #[cfg_attr(rustfmt,rustfmt_skip)]
        let m = Matrix3::new([
            1., 5., 0.,
            -3., 2., 7.,
            0., 6., -3.
        ]);

        assert_eq!(m.submatrix(0, 2), Matrix2::new([-3., 2., 0., 6.]));
        assert_eq!(m.submatrix(2, 2), Matrix2::new([1., 5., -3., 2.]));
    }
    #[test]
    fn minor() {
        #[cfg_attr(rustfmt,rustfmt_skip)]
        let m = Matrix3::new([
            3., 5., 0.,
            2., -1., -7.,
            6., -1., 5.
        ]);

        assert_eq!(m.minor(1, 0), 25.)
    }
    #[test]
    fn cofactor() {
        #[cfg_attr(rustfmt,rustfmt_skip)]
        let m = Matrix3::new([
            3., 5., 0.,
            2., -1., -7.,
            6., -1., 5.
        ]);
        let minor1 = m.minor(0, 0);
        let minor2 = m.minor(1, 0);

        assert_eq!(minor1, -12.);
        assert_eq!(minor1, m.cofactor(0, 0));

        assert_eq!(minor2, 25.0);
        assert_eq!(minor2, -m.cofactor(1, 0));
    }
    #[test]
    fn determinant() {
        #[cfg_attr(rustfmt,rustfmt_skip)]
        let m = Matrix3::new([
            1., 2., 6.,
            -5., 8., -4.,
            2., 6., 4.
        ]);

        assert_eq!(m.cofactor(0, 0), 56.);
        assert_eq!(m.cofactor(0, 1), 12.);
        assert_eq!(m.cofactor(0, 2), -46.);

        assert_eq!(m.determinant(), -196.);
    }
}
