use crate::approx_eq::ApproxEq;
use std::ops;

use super::tuple::Tuple;

#[derive(Debug, Clone, Copy)]
pub struct Matrix2 {
    data: [f64; 4],
}

impl Matrix2 {
    pub fn new(data: [f64; 4]) -> Self {
        Self { data }
    }
    pub fn empty() -> Self {
        Self::new([0.; 4])
    }
    pub fn identiy_matrix() -> Self {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Self::new([
            1., 0.,
            0., 1.,
        ])
    }

    pub fn transpose(&self) -> Self {
        let mut res = self.clone();
        res.data.swap(1, 2);
        res
    }

    pub fn determinant(&self) -> f64 {
        self.data[0] * self.data[3] - self.data[1] * self.data[2]
    }
}

impl PartialEq for Matrix2 {
    fn eq(&self, other: &Matrix2) -> bool {
        self.data
            .iter()
            .enumerate()
            .all(|(id, x)| x.approq_eq(other.data[id]))
    }
}

impl ops::Index<(usize, usize)> for Matrix2 {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        debug_assert!(row < 2);
        debug_assert!(col < 2);
        &self.data[row * 2 + col]
    }
}

impl ops::IndexMut<(usize, usize)> for Matrix2 {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        debug_assert!(row < 2);
        debug_assert!(col < 2);
        &mut self.data[row * 2 + col]
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn determindant() {
        assert_eq!(Matrix2::new([1., 5., -3., 2.]).determinant(), 17.);
    }
}
