use crate::{
    math::{
        matrix::Matrix,
        tuple::{Axis, Tuple},
    },
    scene::{
        animation::{Interpolate, SelfInterpolate},
        object::bounding_box::{Bounded, BoundingBox},
    },
};

use super::{Transform, Transformation, Transformations};

/// An object that can be transformed via LocalTransformation
pub trait LocalTransform: Transform + Bounded {
    fn local_transform(&mut self, t: &LocalTransformations) {
        let matrix = t.into_matrix(self);
        self.transform(&matrix);
    }
}

impl<T: Transform + Bounded> LocalTransform for T {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocalTransformation {
    /// Translates the object so it's center is at origin
    Center,
    /// Translates the object so it's bounding box is above at the axis
    TranslateAbove(Axis),
    /// Translates the object so it's bounding box is below at the axis
    TranslateBelow(Axis),
    /// Scales the object so that it's length along the axis is 1
    NormalizeToAxis(Axis),
    /// Scales the object so that it's bounding box is a cube of side 1
    NormalizeAllAxes,
    /// Scales the object uniformly in all axes such that lenght of the longest side becomes 1
    NormalizeToLongestAxis,
    /// Transformation that maintains the center of the object
    Local(Transformation),
    /// Regular transformation
    Transformation(Transformation),
}

impl<T> Interpolate<T, Vec<Transformation>> for LocalTransformation
where
    T: LocalTransform,
{
    fn interpolated_with(&self, local_obj: &T, at: f64) -> Vec<Transformation> {
        let transformations = self.into_transformations(local_obj);
        transformations.interpolated(at)
    }
}

impl LocalTransformation {
    // Local transform is just translating to origin, transforming and then translating back
    fn local_transform(
        &self,
        bbox: &BoundingBox,
        transform: Transformation,
    ) -> Vec<Transformation> {
        let mut res = LocalTransformation::Center.into_transformations(bbox);
        let put_back = res[0].interpolated(-1.);
        res.push(transform);
        res.push(put_back);
        res
    }
    // Because pivoting is a few transformations combined and the fact that matrices are not
    // interpolatable we are opt to return a type which is interpolatable
    pub fn into_transformations<T: LocalTransform>(&self, local_obj: &T) -> Vec<Transformation> {
        let bbox = local_obj.bounding_box();
        match self {
            Self::Center => {
                let center = bbox.center();
                vec![Transformation::Translation(
                    -center.x(),
                    -center.y(),
                    -center.z(),
                )]
            }
            Self::TranslateAbove(axis) => {
                vec![match axis {
                    Axis::X => Transformation::Translation(-bbox.min.x(), 0., 0.),
                    Axis::Y => Transformation::Translation(0., -bbox.min.y(), 0.),
                    Axis::Z => Transformation::Translation(0., 0., -bbox.min.z()),
                }]
            }
            Self::TranslateBelow(axis) => {
                vec![match axis {
                    Axis::X => Transformation::Translation(-bbox.max.x(), 0., 0.),
                    Axis::Y => Transformation::Translation(0., -bbox.max.y(), 0.),
                    Axis::Z => Transformation::Translation(0., 0., -bbox.max.z()),
                }]
            }
            Self::NormalizeToAxis(axis) => {
                let size = bbox.size();
                let len = match axis {
                    Axis::X => size.x(),
                    Axis::Y => size.y(),
                    Axis::Z => size.z(),
                };
                self.local_transform(bbox, Transformation::scaling_uniform(1. / len))
            }
            Self::NormalizeToLongestAxis => {
                let (_, len) = bbox.longest_axis();
                self.local_transform(bbox, Transformation::scaling_uniform(1. / len))
            }
            Self::NormalizeAllAxes => {
                let size = bbox.size();
                self.local_transform(
                    bbox,
                    Transformation::Scaling(1. / size.x(), 1. / size.y(), 1. / size.z()),
                )
            }

            Self::Local(t) => {
                let mut res = LocalTransformation::Center.into_transformations(bbox);
                let put_back = res[0].interpolated(-1.);
                res.push(*t);
                res.push(put_back);
                res
            }
            Self::Transformation(t) => vec![*t],
        }
    }
    pub fn matrix_at<T: LocalTransform>(&self, local_obj: &T, at: f64) -> Matrix {
        self.interpolated_with(local_obj, at).into()
    }

    pub fn into_matrix<T: LocalTransform>(&self, local_obj: &T) -> Matrix {
        self.into_transformations(local_obj).into()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalTransformations {
    data: Vec<LocalTransformation>,
}

impl LocalTransformations {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    pub fn with_vec(vec: Vec<LocalTransformation>) -> Self {
        Self { data: vec }
    }
    pub fn vec(&self) -> &[LocalTransformation] {
        &self.data
    }
    pub fn push(&mut self, t: LocalTransformation) {
        self.data.push(t);
    }
    pub fn extend(&mut self, other: &Self) {
        self.data.extend(other.data.iter().copied());
    }

    pub fn into_matrix<T: Bounded>(&self, obj: &T) -> Matrix {
        self.interpolated_with(obj, 1.).into()
    }
}

impl TryFrom<LocalTransformations> for Transformations {
    type Error = ();

    fn try_from(value: LocalTransformations) -> Result<Self, Self::Error> {
        value
            .data
            .into_iter()
            .map(|t| match t {
                LocalTransformation::Transformation(t) => Ok(t),
                _ => Err(()),
            })
            .collect::<Result<Vec<Transformation>, Self::Error>>()
            .map(Into::into)
    }
}

impl From<Transformation> for LocalTransformation {
    fn from(value: Transformation) -> Self {
        Self::Transformation(value)
    }
}

impl From<Transformations> for LocalTransformations {
    fn from(value: Transformations) -> Self {
        Self::from(value.data)
    }
}

impl From<Vec<Transformation>> for LocalTransformations {
    fn from(value: Vec<Transformation>) -> Self {
        Self::with_vec(
            value
                .into_iter()
                .map(Into::<LocalTransformation>::into)
                .collect(),
        )
    }
}

impl From<Vec<LocalTransformation>> for LocalTransformations {
    fn from(value: Vec<LocalTransformation>) -> Self {
        Self::with_vec(value)
    }
}

impl<T: Bounded> Interpolate<T, Transformations> for LocalTransformations {
    fn interpolated_with(&self, with: &T, at: f64) -> Transformations {
        let mut bbox = with.bounding_box().clone();
        let mut buf_matrix = Matrix::identity();

        // Because the local transformations depend on the current state of the object, we simulate
        // transformations done to it. To optimize, we limit the transformations of bounding boxes,
        // and instead just transform the matrix, and lazily calculate the actual bbox only upon
        // encountering a local transform.
        Transformations::with_vec(
            self.data
                .iter()
                .flat_map(|t| {
                    match t {
                        LocalTransformation::Transformation(_) => {}
                        _ => {
                            bbox.transform(&buf_matrix);
                            buf_matrix = Matrix::identity();
                        }
                    }
                    let res = t.interpolated_with(&bbox, at);
                    let m = Matrix::from(&res[..]);
                    buf_matrix.transform(&m);

                    res
                })
                .collect(),
        )
    }
}

impl Default for LocalTransformations {
    fn default() -> Self {
        Self::new()
    }
}
#[cfg(test)]
mod local_transform_tests {
    use crate::math::approx_eq::ApproxEq;
    use std::f64::consts::FRAC_PI_3;

    use super::*;
    use crate::{
        assert_approx_eq_low_prec,
        math::{point::Point, vector::Vector},
        scene::object::{bounding_box::BoundingBox, primitive::shape::Shape},
    };

    fn bbox(min: (f64, f64, f64), max: (f64, f64, f64)) -> BoundingBox {
        BoundingBox {
            min: Point::new(min.0, min.1, min.2),
            max: Point::new(max.0, max.1, max.2),
        }
    }

    #[test]
    fn center_translation_moves_center_to_origin() {
        let mut obj = bbox((1., 2., 3.), (5., 6., 7.));

        let transform = LocalTransformation::Center.matrix_at(&obj, 1.0);
        obj.transform(&transform);

        let new_center = obj.center();
        assert_approx_eq_low_prec!(new_center, Point::zero());
    }

    #[test]
    fn translate_above_moves_min_to_origin() {
        let obj = bbox((1., 2., 3.), (5., 6., 7.));

        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let mut obj_copy = obj.clone();
            let transform = LocalTransformation::TranslateAbove(axis).matrix_at(&obj_copy, 1.0);
            obj_copy.transform(&transform);

            let new_min = obj_copy.min;
            match axis {
                Axis::X => assert_approx_eq_low_prec!(new_min.x(), 0.),
                Axis::Y => assert_approx_eq_low_prec!(new_min.y(), 0.),
                Axis::Z => assert_approx_eq_low_prec!(new_min.z(), 0.),
            }
        }
    }

    #[test]
    fn translate_below_moves_max_to_origin() {
        let obj = bbox((1., 2., 3.), (5., 6., 7.));

        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let mut obj_copy = obj.clone();
            let transform = LocalTransformation::TranslateBelow(axis).matrix_at(&obj_copy, 1.0);
            obj_copy.transform(&transform);

            let new_max = obj_copy.max;
            match axis {
                Axis::X => assert_approx_eq_low_prec!(new_max.x(), 0.),
                Axis::Y => assert_approx_eq_low_prec!(new_max.y(), 0.),
                Axis::Z => assert_approx_eq_low_prec!(new_max.z(), 0.),
            }
        }
    }

    #[test]
    fn normalize_to_longest_axis_scales_longest_side_to_one() {
        let mut obj = bbox((0., 0., 0.), (2., 4., 8.));
        let transform = LocalTransformation::NormalizeToLongestAxis.matrix_at(&obj, 1.0);
        obj.transform(&transform);

        let size = obj.bounding_box().size();
        assert_eq!(size, Vector::new(1. / 4., 1. / 2., 1.));
    }

    #[test]
    fn normalize_all_axes_scales_all_sides_to_one() {
        let mut obj = bbox((0., 0., 0.), (2., 4., 8.));
        let transform = LocalTransformation::NormalizeAllAxes.matrix_at(&obj, 1.0);
        obj.transform(&transform);

        let size = obj.size();
        assert_approx_eq_low_prec!(size.x(), 1.0);
        assert_approx_eq_low_prec!(size.y(), 1.0);
        assert_approx_eq_low_prec!(size.z(), 1.0);
    }

    #[test]
    fn normalize_to_axis() {
        let obj = bbox((0., 0., 0.), (2., 4., 8.));

        let axis_to_expected_factor = [(Axis::X, 0.5), (Axis::Y, 0.25), (Axis::Z, 0.125)];

        let size = Vector::new(2., 4., 8.);
        for (axis, factor) in axis_to_expected_factor {
            let mut obj_copy = obj.clone();
            let transform = LocalTransformation::NormalizeToAxis(axis).matrix_at(&obj_copy, 1.0);
            obj_copy.transform(&transform);

            let expected_size = size * factor;
            assert_eq!(obj_copy.size(), expected_size, "Failed for axis: {axis:?}",);
        }
    }
    #[test]
    fn local_rotate_90_degree_multiple_keeps_bounding_box() {
        let obj = bbox((1., 2., 3.), (5., 6., 7.));
        let original_bbox = obj.clone();

        let deg_90 = LocalTransformation::Local(Transformation::Rotation(
            Axis::Y,
            std::f64::consts::FRAC_PI_2,
        ))
        .matrix_at(&obj, 1.0);
        let deg_45 = LocalTransformation::Local(Transformation::Rotation(
            Axis::Y,
            std::f64::consts::FRAC_PI_4,
        ))
        .matrix_at(&obj, 1.0);

        assert_eq!(obj.transform_new(&deg_90).bounding_box(), &original_bbox);
        assert_ne!(obj.transform_new(&deg_45).bounding_box(), &original_bbox);
    }
    #[test]
    fn local_rotate_rotates_around_local_axis() {
        let translate = Matrix::translation(1., 2., 3.);
        let rad = std::f64::consts::FRAC_PI_4;

        let mut obj = Shape::Cube.bounding_box();
        let mut expected = obj.clone();

        obj.transform(&translate);
        obj.transform(
            &LocalTransformation::Local(Transformation::Rotation(Axis::X, rad))
                .matrix_at(&obj, 1.0),
        );

        expected.rotate_x(rad);
        expected.transform(&translate);

        assert_eq!(obj, expected);
    }

    #[test]
    fn transformation_variant_applies_directly() {
        let mut obj = bbox((1., 2., 3.), (4., 5., 6.));
        let transform =
            LocalTransformation::Transformation(Transformation::Translation(1., 2., 3.))
                .matrix_at(&obj, 1.0);
        obj.transform(&transform);

        let new_min = obj.min;
        let new_max = obj.max;

        assert_approx_eq_low_prec!(new_min, Point::new(2., 4., 6.));
        assert_approx_eq_low_prec!(new_max, Point::new(5., 7., 9.));
    }

    #[test]
    fn local_transformations_keep_center() {
        let obj = bbox((1., 2., 3.), (4., 5., 6.));

        let transforms = LocalTransformations::from(vec![
            LocalTransformation::Local(Transformation::Scaling(-1., 2., -3.)),
            LocalTransformation::Local(Transformation::Rotation(Axis::X, FRAC_PI_3)),
            LocalTransformation::Local(Transformation::Shearing(1., -2., 3., -4., 5., 6.)),
        ]);

        let transformed = obj.transform_new(&(&transforms.interpolated_with(&obj, 1.)).into());

        assert_eq!(obj.center(), transformed.center());
    }

    #[test]
    fn local_transformations() {
        let bbox = bbox((0., 0., 0.), (2., 4., 8.));

        let local_transformations = LocalTransformations::from(vec![
            LocalTransformation::Center,
            LocalTransformation::Transformation(Transformation::Translation(1., 2., 3.)),
            LocalTransformation::Local(Transformation::Rotation(
                Axis::X,
                std::f64::consts::FRAC_PI_2,
            )),
            LocalTransformation::NormalizeToLongestAxis,
            LocalTransformation::NormalizeAllAxes,
            LocalTransformation::TranslateAbove(Axis::X),
            LocalTransformation::TranslateBelow(Axis::Y),
            LocalTransformation::Local(Transformation::Scaling(1., -2., 3.)),
        ]);
        let expected = Transformations::from(vec![
            Transformation::Translation(-1., -2., -4.),
            Transformation::Translation(1., 2., 3.),
            Transformation::Translation(-1., -2., -3.),
            Transformation::Rotation(Axis::X, std::f64::consts::FRAC_PI_2),
            Transformation::Translation(1., 2., 3.),
            Transformation::Translation(-1., -2., -3.),
            Transformation::scaling_uniform(1. / 8.),
            Transformation::Translation(1., 2., 3.),
            Transformation::Translation(-1., -2., -3.),
            Transformation::Scaling(4., 1., 2.),
            Transformation::Translation(1., 2., 3.),
            Transformation::Translation(-0.5, 0., 0.),
            Transformation::Translation(0., -2.5, 0.),
            Transformation::Translation(-0.5, 0.5, -3.),
            Transformation::Scaling(1., -2., 3.),
            Transformation::Translation(0.5, -0.5, 3.),
        ]);
        assert_eq!(local_transformations.interpolated_with(&bbox, 1.), expected);
    }
}
