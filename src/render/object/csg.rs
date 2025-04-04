use crate::{
    primitive::matrix::{Matrix, Transform},
    render::{
        intersection::{IntersectionCollection, IntersectionCollector},
        ray::Ray,
    },
};

use super::Object;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeftRight {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsgIntersectionLocation {
    pub hit: LeftRight,
    pub inside_left: bool,
    pub inside_right: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq)]
pub struct CsgObject {
    pub operation: CsgOperation,
    pub left: Object,
    pub right: Object,
}

impl CsgObject {
    fn filter_intersections(&self, xs: &mut IntersectionCollection) {
        *xs.hit_mut() = None; // we might remove the hit
        xs.sort(); // the intersections have to be sorted to easily determine the `inside_*`
        let mut inside_left = false;
        let mut inside_right = false;
        xs.vec_mut().retain(|i| {
            let hit_left = self.left.includes(i.object());
            if hit_left {
                inside_left = !inside_left;
            } else {
                inside_right = !inside_right;
            }
            let location = CsgIntersectionLocation {
                hit: if hit_left {
                    LeftRight::Left
                } else {
                    LeftRight::Right
                },
                inside_left,
                inside_right,
            };
            self.operation.is_intersection_allowed(location)
        });
        // note: the collection is temporary so we don't care about calculating hit
    }
    pub fn intersect<'a>(&'a self, world_ray: &Ray, collector: &mut IntersectionCollector<'a>) {
        let mut temp_collector = IntersectionCollector::default();
        self.left.intersect(world_ray, &mut temp_collector);
        self.right.intersect(world_ray, &mut temp_collector);
        let mut collection =
            IntersectionCollection::from_collector(world_ray.clone(), temp_collector);
        self.filter_intersections(&mut collection);
        collection.into_vec().into_iter().for_each(|i| {
            collector.push(i);
        });
    }
}

impl Transform for CsgObject {
    fn transform(&mut self, matrix: &Matrix) {
        self.left.transform(matrix);
        self.right.transform(matrix);
    }
}

#[cfg(test)]
mod tests {
    use crate::render::intersection::Intersection;
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
    #[test]
    fn filter_intersections() {
        let expected = [
            (CsgOperation::Union, [0., 3.]),
            (CsgOperation::Intersection, [1., 2.]),
            (CsgOperation::Difference, [0., 1.]),
        ];
        expected.iter().copied().for_each(|(operation, expected)| {
            let csg = CsgObject {
                operation,
                left: Object::primitive_with_shape(Shape::Sphere),
                right: Object::primitive_with_shape(Shape::Cube),
            };
            let sphere = &csg.left;
            let cube = &csg.right;
            let mut collection = IntersectionCollection::new(
                Ray::default(),
                vec![
                    Intersection::new(0., sphere),
                    Intersection::new(1., cube),
                    Intersection::new(2., sphere),
                    Intersection::new(3., cube),
                ],
                None,
                true,
            );
            csg.filter_intersections(&mut collection);
            let times = collection
                .vec()
                .iter()
                .map(Intersection::time)
                .collect::<Vec<_>>();
            assert_eq!(times, expected);
        })
    }
}
