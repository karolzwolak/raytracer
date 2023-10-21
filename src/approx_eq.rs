pub const EPSILON: f64 = 1.0e-5;

pub trait ApproxEq<Rhs = Self> {
    fn approq_eq(self, rhs: Rhs) -> bool;
}

impl ApproxEq for f64 {
    fn approq_eq(self, rhs: Self) -> bool {
        (self - rhs).abs() < EPSILON
    }
}
