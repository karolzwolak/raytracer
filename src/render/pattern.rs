use crate::primitive::point::Point;

use super::color::Color;

pub enum Pattern {
    /// Stripe alternating as x changes
    Stripe(Color, Color),
}

impl Pattern {
    pub fn color_at(&self, point: &Point) -> Color {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::primitive::tuple::Tuple;

    use super::*;

    #[test]
    fn stripe_pattern_const_in_y() {
        let stripe = Pattern::Stripe(Color::white(), Color::black());

        assert_eq!(stripe.color_at(&Point::new(0., 0., 0.)), Color::white());
        assert_eq!(stripe.color_at(&Point::new(0., 1., 0.)), Color::white());
        assert_eq!(stripe.color_at(&Point::new(0., 2., 0.)), Color::white());
    }

    #[test]
    fn stripe_pattern_const_in_z() {
        let stripe = Pattern::Stripe(Color::white(), Color::black());

        assert_eq!(stripe.color_at(&Point::new(0., 0., 0.)), Color::white());
        assert_eq!(stripe.color_at(&Point::new(0., 0., 1.)), Color::white());
        assert_eq!(stripe.color_at(&Point::new(0., 0., 2.)), Color::white());
    }

    #[test]
    fn stripe_pattern_alternate_in_x() {
        let black = Color::black();
        let white = Color::white();
        let stripe = Pattern::Stripe(white, black);

        assert_eq!(stripe.color_at(&Point::new(0., 0., 0.)), white);
        assert_eq!(stripe.color_at(&Point::new(0.9, 0., 0.)), white);
        assert_eq!(stripe.color_at(&Point::new(1., 0., 0.)), black);
        assert_eq!(stripe.color_at(&Point::new(-0.1, 0., 0.)), black);
        assert_eq!(stripe.color_at(&Point::new(-1., 0., 0.)), black);
        assert_eq!(stripe.color_at(&Point::new(-1.1, 0., 0.)), white);
    }
}