pub const EPSILON: f64 = 1.0e-8;
pub const LOW_PREC_EPSILON: f64 = 1.0e-4;

pub trait ApproxEq<Rhs = Self> {
    fn approx_eq_epsilon(&self, rhs: &Rhs, epsilon: f64) -> bool;

    fn approx_eq(&self, rhs: &Rhs) -> bool {
        self.approx_eq_epsilon(rhs, EPSILON)
    }

    fn approx_eq_low_prec(&self, rhs: &Rhs) -> bool {
        self.approx_eq_epsilon(rhs, LOW_PREC_EPSILON)
    }
}

impl ApproxEq for f64 {
    fn approx_eq_epsilon(&self, rhs: &Self, epsilon: f64) -> bool {
        self == rhs || (*self - *rhs).abs() < epsilon
    }
}

#[macro_export]
macro_rules! assert_approx_eq_low_prec {
    ($left:expr, $right:expr) => {
        match (&$left, &$right) {
            (left, right) => {
                if !left.approx_eq_low_prec(right) {
                    panic!(
                        "assertion failed: `(left ≈ right)`\n  left: `{:?}`,\n right: `{:?}`",
                        left, right
                    );
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_eq_f64() {
        assert!(0f64.approx_eq(&0f64));
        assert!(!0f64.approx_eq(&EPSILON));
        assert!(!0f64.approx_eq(&-EPSILON));

        assert!(!0f64.approx_eq(&(EPSILON - LOW_PREC_EPSILON)));
        assert_approx_eq_low_prec!(0., 0.5 * LOW_PREC_EPSILON);
    }
}
