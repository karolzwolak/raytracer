use super::Object;

#[derive(Debug, Clone, Copy)]
pub enum LeftRight {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy)]
pub struct CsgIntersectionLocation {
    pub hit: LeftRight,
    pub inside_left: bool,
    pub inside_right: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum CsgOperation {
    Union,
    Intersection,
    Difference,
}

impl CsgOperation {
    fn is_intersection_allowed(&self, l: CsgIntersectionLocation) -> bool {
        match self {
            CsgOperation::Union => match l.hit {
                LeftRight::Left => !l.inside_right,
                LeftRight::Right => !l.inside_left,
            },
            CsgOperation::Intersection => match l.hit {
                LeftRight::Left => l.inside_right,
                LeftRight::Right => l.inside_left,
            },
            CsgOperation::Difference => match l.hit {
                LeftRight::Left => !l.inside_right,
                LeftRight::Right => l.inside_left,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct CsgObject {
    pub operation: CsgOperation,
    pub left: Object,
    pub right: Object,
}

#[cfg(test)]
mod tests {
    use crate::render::object::shape::Shape;

    use super::LeftRight::{Left, Right};
    use super::*;

    #[test]
    fn csg_creation() {
        let _ = CsgObject {
            operation: CsgOperation::Union,
            left: Object::primitive_with_shape(Shape::Sphere),
            right: Object::primitive_with_shape(Shape::Cube),
        };
    }

    fn bool_vec_to_intersection_kind(input: &[bool; 3]) -> CsgIntersectionLocation {
        CsgIntersectionLocation {
            hit: if input[0] { Left } else { Right },
            inside_left: input[1],
            inside_right: input[2],
        }
    }

    fn test_intersection_allowed(operation: CsgOperation, expected: &[([bool; 3], bool)]) {
        expected.iter().for_each(|(input, expected)| {
            let input = bool_vec_to_intersection_kind(input);
            let result = operation.is_intersection_allowed(input);
            assert_eq!(result, *expected, "input: {:?}", input);
        });
    }

    const UNION_ALLOWED_INTERSECTIONS: [([bool; 3], bool); 8] = [
        ([true, true, true], false),
        ([true, true, false], true),
        ([true, false, true], false),
        ([true, false, false], true),
        ([false, true, true], false),
        ([false, true, false], false),
        ([false, false, true], true),
        ([false, false, false], true),
    ];

    #[test]
    fn union_allowed_intersections() {
        test_intersection_allowed(CsgOperation::Union, &UNION_ALLOWED_INTERSECTIONS);
    }
    #[test]
    fn intersection_allowed_intersections() {
        let expected = UNION_ALLOWED_INTERSECTIONS.map(|(input, expected)| (input, !expected));
        test_intersection_allowed(CsgOperation::Intersection, &expected);
    }
    #[test]
    fn difference_allowed_intersections() {
        let expected = [
            ([true, true, true], false),
            ([true, true, false], true),
            ([true, false, true], false),
            ([true, false, false], true),
            ([false, true, true], true),
            ([false, true, false], true),
            ([false, false, true], false),
            ([false, false, false], false),
        ];
        test_intersection_allowed(CsgOperation::Difference, &expected);
    }
}
