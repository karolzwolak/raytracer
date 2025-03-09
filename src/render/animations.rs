use crate::primitive::matrix::{Matrix, Transformation};

#[derive(Debug, Clone, Copy)]
enum AnimationTiming {
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

#[derive(Debug, Clone, Copy)]
enum AnimationDirection {
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

#[derive(Debug, Clone, Copy)]
enum AnimationCount {
    Infinite,
    Count(u32),
}

impl AnimationCount {
    fn still_animate(&self, curr_count: u32) -> bool {
        match self {
            Self::Infinite => true,
            Self::Count(count) => curr_count < *count,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Animation {
    delay: f64,
    duration: f64,
    direction: AnimationDirection,
    timing: AnimationTiming,
    count: AnimationCount,
}

impl Animation {
    fn new(
        delay: f64,
        duration: f64,
        direction: AnimationDirection,
        timing: AnimationTiming,
        count: AnimationCount,
    ) -> Self {
        Self {
            delay,
            duration,
            direction,
            timing,
            count,
        }
    }
}

#[derive(Debug, Clone)]
struct AnimationState {
    animation: Animation,
    time: f64,
    curr_count: u32,
}

impl AnimationState {
    fn new(animation: Animation) -> Self {
        Self {
            animation,
            time: 0.0,
            curr_count: 0,
        }
    }
    fn update(&mut self, dt: f64) -> f64 {
        self.time += dt;
        if self.time < self.animation.delay || !self.animation.count.still_animate(self.curr_count)
        {
            return 0.;
        }
        let mut time = self.time - self.animation.delay;
        if time > self.animation.duration {
            self.curr_count += 1;
            time -= self.animation.duration;
        }
        let normalized_time = time / self.animation.duration;
        let fraction = self.animation.timing.apply(normalized_time);

        self.animation.direction.apply(fraction, self.curr_count)
    }
}

#[derive(Debug, Clone)]
struct TransformAnimation {
    animation_state: AnimationState,
    transformations: Vec<Transformation>,
}

impl TransformAnimation {
    fn new(animation: Animation, transformations: Vec<Transformation>) -> Self {
        Self {
            animation_state: AnimationState::new(animation),
            transformations,
        }
    }

    fn interpolate(&self, factor: f64) -> Matrix {
        self.transformations
            .iter()
            .copied()
            .fold(Matrix::identity(), |acc, t| acc * Matrix::from(t * factor))
    }

    fn update(&mut self, dt: f64) -> Matrix {
        let fraction = self.animation_state.update(dt);
        self.interpolate(fraction)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::{self};

    use crate::primitive::tuple::Axis;

    use super::*;

    const NORMAL_ANIMATION: Animation = Animation {
        delay: 0.0,
        duration: 1.0,
        direction: AnimationDirection::Normal,
        timing: AnimationTiming::Linear,
        count: AnimationCount::Infinite,
    };

    const TRANSFORMATIONS: [Transformation; 5] = [
        Transformation::Rotation(Axis::Z, f64::consts::FRAC_PI_2),
        Transformation::Translation(1.0, 0.0, -5.5),
        Transformation::Rotation(Axis::Y, f64::consts::FRAC_PI_4),
        Transformation::Scaling(0.5, 0.1, 2.),
        Transformation::Rotation(Axis::X, f64::consts::FRAC_PI_6),
    ];

    fn full_transformation() -> Matrix {
        TRANSFORMATIONS
            .iter()
            .fold(Matrix::identity(), |acc, t| acc * Matrix::from(*t))
    }

    #[test]
    fn linear_timing() {
        let linear = AnimationTiming::Linear;
        assert_eq!(linear.apply(0.5), 0.5);
    }

    #[test]
    fn finite_animation_count() {
        let count = AnimationCount::Count(3);

        assert!(count.still_animate(0));
        assert!(count.still_animate(1));
        assert!(count.still_animate(2));
        assert!(!count.still_animate(3));
    }

    #[test]
    fn infinite_animation_count() {
        let count = AnimationCount::Infinite;

        assert!(count.still_animate(0));
        assert!(count.still_animate(u32::MAX));
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
            AnimationCount::Infinite,
        );
        let state = AnimationState::new(animation);

        assert_eq!(state.clone().update(0.5), 0.0);
        assert_eq!(state.clone().update(1.5), 0.5);
    }
}
