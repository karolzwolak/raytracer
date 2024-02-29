pub const EPSILON: f64 = 1.0e-5;
pub const LOW_PREC_EPSILON: f64 = EPSILON * 10.;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_eq_f64() {
        assert!(0f64.approx_eq(&0f64));
        assert!(!0f64.approx_eq(&EPSILON));
        assert!(!0f64.approx_eq(&-EPSILON));

        assert!(!0f64.approx_eq(&(EPSILON - LOW_PREC_EPSILON)));
    }
}
