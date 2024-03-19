use crate::{
    approx_eq::ApproxEq,
    primitive::{matrix::Matrix, point::Point, tuple::Tuple},
};

use super::{color::Color, object::Object};

#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    /// Stripe alternating as x changes
    Stripe {
        c1: Color,
        c2: Color,
        inv_transform: Matrix,
    },
    /// Linear gradient changing in x direction
    Gradient {
        c_start: Color,
        c_dist: Color,
        inv_transform: Matrix,
    },
    /// Ring pattern extending in x and z
    Ring {
        c1: Color,
        c2: Color,
        inv_transform: Matrix,
    },
    /// 3D checkerboard
    Checkers {
        c1: Color,
        c2: Color,
        inv_transform: Matrix,
    },
    /// Pattern that returns points coordinates as color
    TestPattern {
        inv_transform: Matrix,
    },
    Const(Color),
}

impl Pattern {
    pub fn stripe(c1: Color, c2: Color, transform: Option<Matrix>) -> Self {
        Self::Stripe {
            c1,
            c2,
            inv_transform: transform.unwrap_or_default().inverse().unwrap(),
        }
    }

    pub fn gradient(c1: Color, c2: Color, transform: Option<Matrix>) -> Self {
        Self::Gradient {
            c_start: c1,
            c_dist: c2 - c1,
            inv_transform: transform.unwrap_or_default().inverse().unwrap(),
        }
    }

    pub fn ring(c1: Color, c2: Color, transform: Option<Matrix>) -> Self {
        Self::Ring {
            c1,
            c2,
            inv_transform: transform.unwrap_or_default().inverse().unwrap(),
        }
    }

    pub fn checkers(c1: Color, c2: Color, transform: Option<Matrix>) -> Self {
        Self::Checkers {
            c1,
            c2,
            inv_transform: transform.unwrap_or_default().inverse().unwrap(),
        }
    }

    pub fn test_pattern(transform: Option<Matrix>) -> Self {
        Self::TestPattern {
            inv_transform: transform.unwrap_or_default().inverse().unwrap(),
        }
    }

    pub fn color_at(&self, point: &Point) -> Color {
        match self {
            Pattern::Stripe { c1, c2, .. } => {
                if (point.x().floor() % 2.).approx_eq(&0.) {
                    *c1
                } else {
                    *c2
                }
            }

            Pattern::Gradient {
                c_start, c_dist, ..
            } => *c_start + *c_dist * (point.x() - point.x().floor()),

            Pattern::Ring { c1, c2, .. } => {
                let val = (point.x().powi(2) + point.z().powi(2)).sqrt().floor();
                if (val % 2.).approx_eq(&0.) {
                    *c1
                } else {
                    *c2
                }
            }

            Pattern::Checkers { c1, c2, .. } => {
                let x = point.x().floor();
                let y = point.y().floor();
                let z = point.z().floor();

                if ((x + y + z) % 2.).approx_eq(&0.) {
                    *c1
                } else {
                    *c2
                }
            }
            Pattern::Const(c) => *c,
            Pattern::TestPattern { .. } => Color::new(point.x(), point.y(), point.z()),
        }
    }

    pub fn color_at_object(&self, object: &Object, point: Point) -> Color {
        let pattern_point = match self {
            Self::Const(_) => point,

            Self::Stripe { inv_transform, .. }
            | Self::Gradient { inv_transform, .. }
            | Self::Ring { inv_transform, .. }
            | Self::Checkers { inv_transform, .. }
            | Self::TestPattern { inv_transform } => {
                let object_point = object.transformation_inverse() * point;
                *inv_transform * object_point
            }
        };

        self.color_at(&pattern_point)
    }
}

#[cfg(test)]
mod tests {
    use crate::approx_eq::ApproxEq;
    use crate::assert_approx_eq_low_prec;
    use crate::{primitive::tuple::Tuple, render::object::shape::Shape};

    use super::*;

    #[test]
    fn stripe_pattern_const_in_y() {
        let stripe = Pattern::stripe(Color::white(), Color::black(), None);

        assert_approx_eq_low_prec!(stripe.color_at(&Point::new(0., 0., 0.)), Color::white());
        assert_approx_eq_low_prec!(stripe.color_at(&Point::new(0., 1., 0.)), Color::white());
        assert_approx_eq_low_prec!(stripe.color_at(&Point::new(0., 2., 0.)), Color::white());
    }

    #[test]
    fn stripe_pattern_const_in_z() {
        let stripe = Pattern::stripe(Color::white(), Color::black(), None);

        assert_approx_eq_low_prec!(stripe.color_at(&Point::new(0., 0., 0.)), Color::white());
        assert_approx_eq_low_prec!(stripe.color_at(&Point::new(0., 0., 1.)), Color::white());
        assert_approx_eq_low_prec!(stripe.color_at(&Point::new(0., 0., 2.)), Color::white());
    }

    #[test]
    fn stripe_pattern_alternate_in_x() {
        let black = Color::black();
        let white = Color::white();
        let stripe = Pattern::stripe(white, black, None);

        assert_approx_eq_low_prec!(stripe.color_at(&Point::new(0., 0., 0.)), white);
        assert_approx_eq_low_prec!(stripe.color_at(&Point::new(0.9, 0., 0.)), white);
        assert_approx_eq_low_prec!(stripe.color_at(&Point::new(1., 0., 0.)), black);
        assert_approx_eq_low_prec!(stripe.color_at(&Point::new(-0.1, 0., 0.)), black);
        assert_approx_eq_low_prec!(stripe.color_at(&Point::new(-1., 0., 0.)), black);
        assert_approx_eq_low_prec!(stripe.color_at(&Point::new(-1.1, 0., 0.)), white);
    }

    #[test]
    fn stripes_with_object_transformation() {
        let sphere =
            Object::primitive_with_transformation(Shape::Sphere, Matrix::scaling(2., 2., 2.));
        let stripe = Pattern::stripe(Color::white(), Color::black(), None);

        assert_approx_eq_low_prec!(
            stripe.color_at_object(&sphere, Point::new(1.5, 0., 0.)),
            Color::white()
        );
    }

    #[test]
    fn stripes_with_pattern_transformation() {
        let sphere = Object::primitive_with_shape(Shape::Sphere);
        let stripe = Pattern::stripe(
            Color::white(),
            Color::black(),
            Some(Matrix::scaling_uniform(2.)),
        );

        assert_approx_eq_low_prec!(
            stripe.color_at_object(&sphere, Point::new(1.5, 0., 0.)),
            Color::white()
        );
    }

    #[test]
    fn stripes_with_object_and_pattern_transformation() {
        let sphere =
            Object::primitive_with_transformation(Shape::Sphere, Matrix::scaling_uniform(2.));
        let stripe = Pattern::stripe(
            Color::white(),
            Color::black(),
            Some(Matrix::translation(0.5, 0., 0.)),
        );

        assert_approx_eq_low_prec!(
            stripe.color_at_object(&sphere, Point::new(2.5, 0., 0.)),
            Color::white()
        );
    }

    #[test]
    fn gradient_linearly_interpolates_between_colors() {
        let gradient = Pattern::gradient(Color::white(), Color::black(), None);

        assert_approx_eq_low_prec!(gradient.color_at(&Point::new(0., 0., 0.)), Color::white());
        assert_approx_eq_low_prec!(
            gradient.color_at(&Point::new(0.25, 0., 0.)),
            Color::new(0.75, 0.75, 0.75)
        );
        assert_approx_eq_low_prec!(
            gradient.color_at(&Point::new(0.5, 0., 0.)),
            Color::new(0.5, 0.5, 0.5)
        );
        assert_approx_eq_low_prec!(
            gradient.color_at(&Point::new(0.75, 0., 0.)),
            Color::new(0.25, 0.25, 0.25)
        );
    }

    #[test]
    fn ring_should_extend_in_both_x_and_z() {
        let ring = Pattern::ring(Color::white(), Color::black(), None);

        assert_approx_eq_low_prec!(ring.color_at(&Point::new(0., 0., 0.)), Color::white());
        assert_approx_eq_low_prec!(ring.color_at(&Point::new(1., 0., 0.)), Color::black());
        assert_approx_eq_low_prec!(ring.color_at(&Point::new(0., 0., 1.)), Color::black());
        assert_approx_eq_low_prec!(ring.color_at(&Point::new(0.708, 0., 0.708)), Color::black());
    }

    #[test]
    fn checkers_should_repeat_in_x() {
        let checkers = Pattern::checkers(Color::white(), Color::black(), None);
        assert_approx_eq_low_prec!(checkers.color_at(&Point::new(0., 0., 0.)), Color::white());
        assert_approx_eq_low_prec!(checkers.color_at(&Point::new(0.99, 0., 0.)), Color::white());
        assert_approx_eq_low_prec!(checkers.color_at(&Point::new(1.01, 0., 0.)), Color::black());
    }

    #[test]
    fn checkers_should_repeat_in_y() {
        let checkers = Pattern::checkers(Color::white(), Color::black(), None);
        assert_approx_eq_low_prec!(checkers.color_at(&Point::new(0., 0., 0.)), Color::white());
        assert_approx_eq_low_prec!(checkers.color_at(&Point::new(0., 0.99, 0.)), Color::white());
        assert_approx_eq_low_prec!(checkers.color_at(&Point::new(0., 1.01, 0.)), Color::black());
    }

    #[test]
    fn checkers_should_repeat_in_z() {
        let checkers = Pattern::checkers(Color::white(), Color::black(), None);
        assert_approx_eq_low_prec!(checkers.color_at(&Point::new(0., 0., 0.)), Color::white());
        assert_approx_eq_low_prec!(checkers.color_at(&Point::new(0., 0., 0.99)), Color::white());
        assert_approx_eq_low_prec!(checkers.color_at(&Point::new(0., 0., 1.01)), Color::black());
    }
}
