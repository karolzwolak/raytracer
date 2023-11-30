use crate::primitive::{matrix4::Matrix4, point::Point, tuple::Tuple};

use super::{color::Color, object::Object};

#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    /// Stripe alternating as x changes
    Stripe {
        c1: Color,
        c2: Color,
        inv_transform: Matrix4,
    },
    Const(Color),
}

impl Pattern {
    pub fn stripe(c1: Color, c2: Color, transform: Option<Matrix4>) -> Self {
        Pattern::Stripe {
            c1,
            c2,
            inv_transform: transform.unwrap_or_default().inverse().unwrap(),
        }
    }
    pub fn color_at(&self, point: &Point) -> Color {
        match self {
            Pattern::Stripe { c1, c2, .. } => {
                if point.x().floor().abs() as usize % 2 == 0 {
                    *c1
                } else {
                    *c2
                }
            }
            Pattern::Const(c) => *c,
        }
    }

    pub fn inv_transformation(&self) -> Matrix4 {
        match self {
            Self::Stripe { inv_transform, .. } => *inv_transform,
            Self::Const { .. } => Matrix4::identity_matrix(),
        }
    }

    pub fn color_at_object(&self, object: &Object, point: Point) -> Color {
        let object_point = object.transformation_inverse().unwrap() * point;
        let pattern_point = self.inv_transformation() * object_point;

        self.color_at(&pattern_point)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        primitive::tuple::Tuple,
        render::object::Shape,
        transformation::{scaling_matrix, translation_matrix},
    };

    use super::*;

    #[test]
    fn stripe_pattern_const_in_y() {
        let stripe = Pattern::stripe(Color::white(), Color::black(), None);

        assert_eq!(stripe.color_at(&Point::new(0., 0., 0.)), Color::white());
        assert_eq!(stripe.color_at(&Point::new(0., 1., 0.)), Color::white());
        assert_eq!(stripe.color_at(&Point::new(0., 2., 0.)), Color::white());
    }

    #[test]
    fn stripe_pattern_const_in_z() {
        let stripe = Pattern::stripe(Color::white(), Color::black(), None);

        assert_eq!(stripe.color_at(&Point::new(0., 0., 0.)), Color::white());
        assert_eq!(stripe.color_at(&Point::new(0., 0., 1.)), Color::white());
        assert_eq!(stripe.color_at(&Point::new(0., 0., 2.)), Color::white());
    }

    #[test]
    fn stripe_pattern_alternate_in_x() {
        let black = Color::black();
        let white = Color::white();
        let stripe = Pattern::stripe(white, black, None);

        assert_eq!(stripe.color_at(&Point::new(0., 0., 0.)), white);
        assert_eq!(stripe.color_at(&Point::new(0.9, 0., 0.)), white);
        assert_eq!(stripe.color_at(&Point::new(1., 0., 0.)), black);
        assert_eq!(stripe.color_at(&Point::new(-0.1, 0., 0.)), black);
        assert_eq!(stripe.color_at(&Point::new(-1., 0., 0.)), black);
        assert_eq!(stripe.color_at(&Point::new(-1.1, 0., 0.)), white);
    }

    #[test]
    fn stripes_with_object_transformation() {
        let sphere = Object::with_transformation(Shape::Sphere, scaling_matrix(2., 2., 2.));
        let stripe = Pattern::stripe(Color::white(), Color::black(), None);

        assert_eq!(
            stripe.color_at_object(&sphere, Point::new(1.5, 0., 0.)),
            Color::white()
        );
    }

    #[test]
    fn stripes_with_pattern_transformation() {
        let sphere = Object::with_shape(Shape::Sphere);
        let stripe = Pattern::stripe(
            Color::white(),
            Color::black(),
            Some(scaling_matrix(2., 2., 2.)),
        );

        assert_eq!(
            stripe.color_at_object(&sphere, Point::new(1.5, 0., 0.)),
            Color::white()
        );
    }

    #[test]
    fn stripes_with_object_and_pattern_transformation() {
        let sphere = Object::with_transformation(Shape::Sphere, scaling_matrix(2., 2., 2.));
        let stripe = Pattern::stripe(
            Color::white(),
            Color::black(),
            Some(translation_matrix(0.5, 0., 0.)),
        );

        assert_eq!(
            stripe.color_at_object(&sphere, Point::new(2.5, 0., 0.)),
            Color::white()
        );
    }
}
