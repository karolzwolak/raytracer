use std::{ops, str::FromStr};

use crate::{
    math::{approx_eq::ApproxEq, matrix::Matrix, transform::local_transform::LocalTransformations},
    scene::object::bounding_box::Bounded,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnimationTiming {
    Ease,
    EaseIn,
    EaseOut,
    EaseInOut,
    Linear,
}

impl AnimationTiming {
    /// Converts value in range [0, 1] to a value in the same range
    fn apply(&self, pos: f64) -> f64 {
        match self {
            Self::Ease => 6. * pos.powi(5) - 15. * pos.powi(4) + 10. * pos.powi(3),
            Self::EaseIn => pos.powi(3),
            Self::EaseOut => 1. - (1. - pos).powi(3),
            Self::EaseInOut => 3. * pos.powi(2) - 2. * pos.powi(3),
            Self::Linear => pos,
        }
    }
}

impl FromStr for AnimationTiming {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ease" => Ok(Self::Ease),
            "ease-in" => Ok(Self::EaseIn),
            "ease-out" => Ok(Self::EaseOut),
            "ease-in-out" => Ok(Self::EaseInOut),
            "linear" => Ok(Self::Linear),
            _ => Err(()),
        }
    }
}

impl Default for AnimationTiming {
    fn default() -> Self {
        Self::Ease
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnimationDirection {
    Normal,
    Reverse,
    Alternate,
    AlternateReverse,
}

impl FromStr for AnimationDirection {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "normal" => Ok(Self::Normal),
            "reverse" => Ok(Self::Reverse),
            "alternate" => Ok(Self::Alternate),
            "alternate-reverse" => Ok(Self::AlternateReverse),
            _ => Err(()),
        }
    }
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
    pub delay: f64,
    pub duration: f64,
    pub direction: AnimationDirection,
    pub timing: AnimationTiming,
    pub repeat: AnimationRepeat,
}

impl Default for Animation {
    fn default() -> Self {
        Self {
            delay: 0.0,
            duration: 1.0,
            direction: AnimationDirection::default(),
            timing: AnimationTiming::default(),
            repeat: AnimationRepeat::default(),
        }
    }
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

    fn repeats_after_iters(&self) -> u32 {
        let count = match self.repeat {
            AnimationRepeat::Infinite => 1,
            AnimationRepeat::Repeat(count) => count,
        };
        let alternates = matches!(
            self.direction,
            AnimationDirection::Alternate | AnimationDirection::AlternateReverse
        );

        if alternates && count % 2 == 1 {
            count * 2
        } else {
            count
        }
    }

    fn effective_period(&self) -> f64 {
        if self.duration <= 0. {
            return 0.;
        }
        let repeats = self.repeats_after_iters();

        self.delay + repeats as f64 * self.duration
    }

    fn val_at(&self, time: f64) -> f64 {
        let time = time - self.delay;
        if time < 0. {
            return 0.;
        }
        let curr_count = (time / self.duration) as u32;
        if !self.repeat.still_animate(curr_count) {
            return 1.;
        }
        let curr_time = time - curr_count as f64 * self.duration;
        let normalized_time = curr_time / self.duration;

        let fraction = self.timing.apply(normalized_time);

        self.direction.apply(fraction, curr_count)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformAnimation {
    animation: Animation,
    transformations: LocalTransformations,
}

impl<T: Bounded> Interpolate<T, Matrix> for TransformAnimation {
    fn interpolated_with(&self, with: &T, at: f64) -> Matrix {
        let at = self.animation.val_at(at);
        if at.approx_eq(&0.) {
            return Matrix::identity();
        }
        self.transformations.interpolated_with(with, at).into()
    }
}

impl TransformAnimation {
    pub fn new(animation: Animation, transformations: LocalTransformations) -> Self {
        Self {
            animation,
            transformations,
        }
    }

    pub fn animations(&self) -> &Animation {
        &self.animation
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Animations {
    vec: Vec<TransformAnimation>,
}

impl<T: Bounded> Interpolate<T, Matrix> for Animations {
    fn interpolated_with(&self, with: &T, at: f64) -> Matrix {
        Matrix::from_iter(self.vec.iter().map(|a| a.interpolated_with(with, at)))
    }
}

fn seconds_to_millis(seconds: f64) -> u32 {
    (seconds * 1000.0).round() as u32
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: u32, b: u32) -> u32 {
    if gcd(a, b) == 0 { 0 } else { a * b / gcd(a, b) }
}

/// Calculates the cycle duration of two animations
/// Zero values are treated as no animation, the cycle duration is the other value
pub fn animation_cycle(a: u32, b: u32) -> u32 {
    match (a, b) {
        (0, _) => b,
        (_, 0) => a,
        _ => lcm(a, b),
    }
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

    pub fn vec(&self) -> &[TransformAnimation] {
        &self.vec
    }

    pub fn cycle_duration_ms(&self) -> u32 {
        self.vec
            .iter()
            .map(|a| seconds_to_millis(a.animations().effective_period()))
            .fold(0, animation_cycle)
    }
}

impl From<Vec<TransformAnimation>> for Animations {
    fn from(vec: Vec<TransformAnimation>) -> Self {
        Self::with_vec(vec)
    }
}

pub trait Base: Sized {
    fn base(&self) -> Self;
}

/// A object that can be interpolated into itself without additional parameters
pub trait SelfInterpolate: Sized {
    fn interpolated(&self, at: f64) -> Self;
}

impl<T> SelfInterpolate for T
where
    T: Interpolate<Self, Self> + Base,
{
    fn interpolated(&self, at: f64) -> Self {
        self.interpolated_with(&self.base(), at)
    }
}

/// A object that can be interpolated, but needs additional parameter
pub trait Interpolate<With, Output>: Sized {
    fn interpolated_with(&self, with: &With, at: f64) -> Output;
}

impl<T> SelfInterpolate for Vec<T>
where
    T: SelfInterpolate,
{
    fn interpolated(&self, at: f64) -> Self {
        self.iter().map(|e| e.interpolated(at)).collect()
    }
}

impl<T> Interpolate<Self, Self> for T
where
    T: Clone
        + ops::Sub<Self, Output = Self>
        + ops::Mul<f64, Output = Self>
        + ops::Add<Self, Output = Self>,
{
    fn interpolated_with(&self, base: &Self, at: f64) -> Self {
        let diff = self.clone() - base.clone();
        base.clone() + diff * at
    }
}

#[cfg(test)]
mod tests {
    use std::f64::{self};

    use crate::{
        math::{
            point::Point,
            transform::{Transform, Transformation, Transformations},
            tuple::Axis,
        },
        scene::object::{
            Object, ObjectKind, PrimitiveObject, bounding_box::BoundingBox, group::ObjectGroup,
        },
    };

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
            Transformations::from(&TRANSFORMATIONS[..]).into(),
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
    fn after_repeat_animations_stays_fully_transformed() {
        let animation = Animation::new(
            1.0,
            1.0,
            AnimationDirection::Normal,
            AnimationTiming::Linear,
            AnimationRepeat::Repeat(2),
        );

        assert_eq!(animation.val_at(3.0), 1.0);
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

        assert_eq!(
            animation.interpolated_with(&BoundingBox::unit(), 0.0),
            Matrix::identity()
        );
    }

    #[test]
    fn full_interpolation_is_next_count() {
        let animation = transform_animation();

        assert_eq!(
            animation.interpolated_with(&BoundingBox::unit(), 1.0),
            Matrix::identity(),
        );
    }
    #[test]
    fn full_interpolation_is_full_transformation() {
        let animation = transform_animation();

        let full = 1.0 - 10e-10; // 1.0 is next animation count
        assert_eq!(
            animation.interpolated_with(&BoundingBox::unit(), full),
            full_transformation()
        );
    }

    #[test]
    fn animation_interpolate() {
        let transforms = Transformations::from(vec![
            Transformation::Translation(12., 12., 12.),
            Transformation::Scaling(2., 2., 2.),
        ])
        .into();
        let animation = TransformAnimation::new(NORMAL_ANIMATION, transforms);

        assert_eq!(
            animation.interpolated_with(&BoundingBox::unit(), 0.5),
            Matrix::translation(6., 6., 6.)
                .scale(1.5, 1.5, 1.5)
                .transformed()
        );
    }

    #[test]
    fn not_normalized_duration() {
        let animation = Animation {
            delay: 0.25,
            duration: 0.5,
            timing: AnimationTiming::Linear,
            ..Default::default()
        };

        assert_eq!(animation.val_at(1.5), 0.5);
    }

    #[test]
    fn animation_cycle_alternate() {
        let alternate = Animation {
            duration: 0.1,
            direction: AnimationDirection::Alternate,
            ..Default::default()
        };
        let normal = Animation {
            duration: 0.15,
            direction: AnimationDirection::Normal,
            ..Default::default()
        };
        let animations = Animations::with_vec(vec![
            TransformAnimation::new(alternate, LocalTransformations::default()),
            TransformAnimation::new(normal, LocalTransformations::default()),
        ]);
        assert_eq!(animations.cycle_duration_ms(), 600);
    }

    #[test]
    fn animation_cycle_alternate_odd_repeat() {
        let alternate = Animation {
            duration: 0.1,
            repeat: AnimationRepeat::Repeat(3),
            direction: AnimationDirection::Alternate,
            ..Default::default()
        };
        let normal = Animation {
            duration: 0.3,
            repeat: AnimationRepeat::Repeat(1),
            direction: AnimationDirection::Normal,
            ..Default::default()
        };
        let animations = Animations::with_vec(vec![
            TransformAnimation::new(alternate, LocalTransformations::default()),
            TransformAnimation::new(normal, LocalTransformations::default()),
        ]);
        assert_eq!(animations.cycle_duration_ms(), 600);
    }

    fn object_with_animation(obj_kind: ObjectKind, animation: Animation) -> Object {
        let anim = TransformAnimation::new(animation, LocalTransformations::default());

        Object::animated(obj_kind, Animations::with_vec(vec![anim]))
    }

    fn primitive_with_animation(animation: Animation) -> Object {
        let sphere = PrimitiveObject::sphere(Point::zero(), 1.);
        let obj_kind = ObjectKind::primitive(sphere);

        object_with_animation(obj_kind, animation)
    }

    fn group_with_animation(animation: Animation, children: Vec<Object>) -> Object {
        let group = ObjectGroup::new(children);
        let obj_kind = ObjectKind::group(group);

        object_with_animation(obj_kind, animation)
    }

    #[test]
    fn animation_cycle_nested_groups() {
        let leaf_animation = Animation {
            duration: 0.12,
            direction: AnimationDirection::AlternateReverse,
            ..Default::default()
        };
        let mid_animation = Animation {
            duration: 0.4,
            direction: AnimationDirection::Alternate,
            repeat: AnimationRepeat::Repeat(2),
            ..Default::default()
        };
        let root_animation = Animation {
            duration: 0.3,
            direction: AnimationDirection::Normal,
            repeat: AnimationRepeat::Repeat(1),
            ..Default::default()
        };
        assert_eq!(leaf_animation.effective_period(), 0.24);
        assert_eq!(mid_animation.effective_period(), 0.8);
        assert_eq!(root_animation.effective_period(), 0.3);

        let leaf = primitive_with_animation(leaf_animation);
        let mid = group_with_animation(mid_animation, vec![leaf]);
        let root = group_with_animation(root_animation, vec![mid]);

        assert_eq!(root.animation_cycle_duration_ms(), 2400);
    }

    #[test]
    fn animation_cycle_redundant_nesting() {
        let a_animation = Animation {
            duration: 0.4,
            direction: AnimationDirection::AlternateReverse,
            ..Default::default()
        };
        let b_animation = Animation {
            duration: 0.3,
            direction: AnimationDirection::Alternate,
            ..Default::default()
        };
        assert_eq!(a_animation.effective_period(), 0.8);
        assert_eq!(b_animation.effective_period(), 0.6);

        let animations = Animations::with_vec(vec![
            TransformAnimation::new(a_animation, LocalTransformations::default()),
            TransformAnimation::new(b_animation, LocalTransformations::default()),
        ]);
        let group = ObjectGroup::new(vec![]);
        let leaf = Object::animated(ObjectKind::group(group), animations);

        let root = Object::group_with_children(vec![leaf.clone(), leaf]);

        assert_eq!(root.animation_cycle_duration_ms(), 2400);
    }

    #[test]
    fn animation_cycle_delays() {
        let animation_a = Animation {
            delay: 0.1,
            duration: 0.2,
            direction: AnimationDirection::Normal,
            repeat: AnimationRepeat::Repeat(2),
            ..Default::default()
        };
        let animation_b = Animation {
            delay: 0.05,
            duration: 0.3,
            direction: AnimationDirection::Normal,
            repeat: AnimationRepeat::Repeat(1),
            ..Default::default()
        };
        let animations = Animations::with_vec(vec![
            TransformAnimation::new(animation_a, LocalTransformations::default()),
            TransformAnimation::new(animation_b, LocalTransformations::default()),
        ]);
        assert_eq!(animations.cycle_duration_ms(), 3500);
    }

    #[test]
    fn animation_ms_rounding() {
        assert_eq!(seconds_to_millis(0.99999), 1000);
    }

    #[test]
    fn animation_cycle_rounding() {
        let third = Animation {
            duration: 0.33333,
            direction: AnimationDirection::Normal,
            repeat: AnimationRepeat::Repeat(3),
            ..Default::default()
        };
        let one = Animation {
            duration: 1.,
            direction: AnimationDirection::Normal,
            ..Default::default()
        };
        let animations = Animations::with_vec(vec![
            TransformAnimation::new(third, LocalTransformations::default()),
            TransformAnimation::new(one, LocalTransformations::default()),
        ]);
        assert_eq!(animations.cycle_duration_ms(), 1000);
    }

    #[test]
    fn animation_cycle_empty() {
        let animations = Animations::empty();
        assert_eq!(animations.cycle_duration_ms(), 0);
    }
}
