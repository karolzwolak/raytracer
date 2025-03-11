use crate::{
    approx_eq::ApproxEq,
    primitive::matrix::{Matrix, TransformationVec},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnimationTiming {
    Linear,
}

impl AnimationTiming {
    /// Converts value in range [0, 1] to a value in the same range
    fn apply(&self, pos: f64) -> f64 {
        match self {
            Self::Linear => pos,
        }
    }
}

impl Default for AnimationTiming {
    fn default() -> Self {
        Self::Linear
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnimationDirection {
    Normal,
    Reverse,
    Alternate,
    AlternateReverse,
}

impl AnimationDirection {
    fn apply(&self, time: f64, curr_count: u32) -> f64 {
        match self {
            Self::Normal => time,
            Self::Reverse => 1. - time,
            Self::Alternate => {
                if curr_count % 2 == 0 {
                    time
                } else {
                    1. - time
                }
            }
            Self::AlternateReverse => {
                if curr_count % 2 == 0 {
                    1. - time
                } else {
                    time
                }
            }
        }
    }
}

impl Default for AnimationDirection {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnimationRepeat {
    Infinite,
    Repeat(u32),
}

impl AnimationRepeat {
    fn still_animate(&self, curr_count: u32) -> bool {
        match self {
            Self::Infinite => true,
            Self::Repeat(count) => curr_count < *count,
        }
    }
}

impl Default for AnimationRepeat {
    fn default() -> Self {
        Self::Infinite
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Animation {
    delay: f64,
    duration: f64,
    direction: AnimationDirection,
    timing: AnimationTiming,
    repeat: AnimationRepeat,
}

impl Animation {
    pub fn new(
        delay: f64,
        duration: f64,
        direction: AnimationDirection,
        timing: AnimationTiming,
        repeat: AnimationRepeat,
    ) -> Self {
        Self {
            delay,
            duration,
            direction,
            timing,
            repeat,
        }
    }
    fn val_at(&self, time: f64) -> f64 {
        let time = time - self.delay;
        if time < 0. {
            return 0.;
        }
        let curr_count = (time / self.duration) as u32;
        let normalized_time = time % self.duration;

        let fraction = self.timing.apply(normalized_time);

        self.direction.apply(fraction, curr_count)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformAnimation {
    animation: Animation,
    transformations: TransformationVec,
}

impl TransformAnimation {
    pub fn new(animation: Animation, transformations: TransformationVec) -> Self {
        Self {
            animation,
            transformations,
        }
    }

    fn interpolate(&self, factor: f64) -> Matrix {
        if factor.approx_eq(&0.) {
            return Matrix::identity();
        }
        Matrix::from(&self.transformations)
    }

    pub fn matrix_at(&self, time: f64) -> Matrix {
        let fraction = self.animation.val_at(time);
        self.interpolate(fraction)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Animations {
    vec: Vec<TransformAnimation>,
}

impl Animations {
    pub fn empty() -> Self {
        Self { vec: Vec::new() }
    }

    pub fn with_vec(vec: Vec<TransformAnimation>) -> Self {
        Self { vec }
    }

    pub fn add(&mut self, animation: TransformAnimation) {
        self.vec.push(animation);
    }

    pub fn matrix_at(&self, dt: f64) -> Matrix {
        Matrix::from_iter(self.vec.iter().map(|a| a.matrix_at(dt)))
    }
}

pub trait Animate {
    fn animate(&mut self, dt: f64);
}

#[cfg(test)]
mod tests {
    use std::f64::{self};

    use crate::primitive::{matrix::Transformation, tuple::Axis};

    use super::*;

    const NORMAL_ANIMATION: Animation = Animation {
        delay: 0.0,
        duration: 1.0,
        direction: AnimationDirection::Normal,
        timing: AnimationTiming::Linear,
        repeat: AnimationRepeat::Infinite,
    };

    const TRANSFORMATIONS: [Transformation; 5] = [
        Transformation::Rotation(Axis::Z, f64::consts::FRAC_PI_2),
        Transformation::Translation(1.0, 0.0, -5.5),
        Transformation::Rotation(Axis::Y, f64::consts::FRAC_PI_4),
        Transformation::Scaling(0.5, 0.1, 2.),
        Transformation::Rotation(Axis::X, f64::consts::FRAC_PI_6),
    ];

    fn full_transformation() -> Matrix {
        Matrix::from(&TRANSFORMATIONS[..])
    }

    fn transform_animation() -> TransformAnimation {
        TransformAnimation::new(
            NORMAL_ANIMATION,
            TransformationVec::from(&TRANSFORMATIONS[..]),
        )
    }

    #[test]
    fn linear_timing() {
        let linear = AnimationTiming::Linear;
        assert_eq!(linear.apply(0.5), 0.5);
    }

    #[test]
    fn finite_animation_repeat() {
        let repeat = AnimationRepeat::Repeat(3);

        assert!(repeat.still_animate(0));
        assert!(repeat.still_animate(1));
        assert!(repeat.still_animate(2));
        assert!(!repeat.still_animate(3));
    }

    #[test]
    fn infinite_animation_repeat() {
        let repeat = AnimationRepeat::Infinite;

        assert!(repeat.still_animate(0));
        assert!(repeat.still_animate(u32::MAX));
    }

    #[test]
    fn animation_directions() {
        let normal = AnimationDirection::Normal;
        let reverse = AnimationDirection::Reverse;
        let alternate = AnimationDirection::Alternate;
        let alternate_reverse = AnimationDirection::AlternateReverse;

        let time = 0.25;
        assert_eq!(normal.apply(time, 0), time);
        assert_eq!(normal.apply(time, 1), time);

        assert_eq!(reverse.apply(time, 0), 1. - time);
        assert_eq!(reverse.apply(time, 1), 1. - time);

        assert_eq!(alternate.apply(time, 0), time);
        assert_eq!(alternate.apply(time, 1), 1. - time);
        assert_eq!(alternate.apply(time, 2), time);

        assert_eq!(alternate_reverse.apply(time, 0), 1. - time);
        assert_eq!(alternate_reverse.apply(time, 1), time);
        assert_eq!(alternate_reverse.apply(time, 2), 1. - time);
    }

    #[test]
    fn animation_delay() {
        let animation = Animation::new(
            1.0,
            1.0,
            AnimationDirection::Normal,
            AnimationTiming::Linear,
            AnimationRepeat::Infinite,
        );

        assert_eq!(animation.val_at(0.5), 0.0);
        assert_eq!(animation.val_at(1.5), 0.5);
    }

    #[test]
    fn zero_interpolation_is_identity() {
        let animation = transform_animation();

        assert_eq!(animation.interpolate(0.0), Matrix::identity());
    }

    #[test]
    fn full_interpolation_is_full_transformation() {
        let animation = transform_animation();

        assert_eq!(animation.interpolate(1.0), full_transformation());
    }
}
