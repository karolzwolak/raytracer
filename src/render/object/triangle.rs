use crate::primitive::{point::Point, vector::Vector};

#[derive(Clone, Debug, PartialEq)]
pub struct Triangle {
    p1: Point,
    p2: Point,
    p3: Point,
    e1: Vector,
    e2: Vector,
    normal: Vector,
}

impl Triangle {
    pub fn new(p1: Point, p2: Point, p3: Point) -> Self {
        let v12 = p2 - p1;
        let v13 = p3 - p1;
        let normal = v13.cross(v12).normalize();

        Self {
            p1,
            p2,
            p3,
            e1: v12,
            e2: v13,
            normal,
        }
    }

    pub fn p1(&self) -> Point {
        self.p1
    }

    pub fn p2(&self) -> Point {
        self.p2
    }

    pub fn p3(&self) -> Point {
        self.p3
    }

    pub fn e1(&self) -> Vector {
        self.e1
    }

    pub fn e2(&self) -> Vector {
        self.e2
    }

    pub fn normal(&self) -> Vector {
        self.normal
    }
}

#[cfg(test)]
mod tests {
    use crate::approx_eq::ApproxEq;
    use crate::{
        primitive::{point::Point, tuple::Tuple, vector::Vector},
        render::{
            object::{shape::Shape, triangle::Triangle, Object},
            ray::Ray,
        },
    };

    #[test]
    fn constructing_triangle() {
        let p1 = Point::new(0., 1., 0.);
        let p2 = Point::new(-1., 0., 0.);
        let p3 = Point::new(1., 0., 0.);

        let t = Triangle::new(p1, p2, p3);

        assert_eq!(t.e1, Vector::new(-1., -1., 0.));
        assert_eq!(t.e2, Vector::new(1., -1., 0.));
        assert_eq!(t.normal, Vector::new(0., 0., -1.));
    }

    fn get_triangle() -> Object {
        Object::with_shape(Shape::triangle(
            Point::new(0., 1., 0.),
            Point::new(-1., 0., 0.),
            Point::new(1., 0., 0.),
        ))
    }

    #[test]
    fn finding_normal_on_triangle() {
        let t = get_triangle();
        let t_shape = match t.shape() {
            Shape::Triangle(ref triangle) => triangle.clone(),
            _ => unreachable!(),
        };

        assert_eq!(t.normal_vector_at(Point::new(1., 0.5, 0.)), t_shape.normal);
        assert_eq!(
            t.normal_vector_at(Point::new(-0.5, 0.75, 0.)),
            t_shape.normal
        );
        assert_eq!(
            t.normal_vector_at(Point::new(0.5, 0.25, 0.)),
            t_shape.normal
        );
    }

    #[test]
    fn intersecting_triangle_with_parallel_ray() {
        let t = get_triangle();

        let ray = Ray::new(Point::new(0., -1., -2.), Vector::new(0., 1., 0.));
        assert!(!t.is_intersected_by_ray(&ray));
    }

    #[test]
    fn ray_misses_p1_p3_edge() {
        let t = get_triangle();
        let ray = Ray::new(Point::new(1., 1., -2.), Vector::new(0., 0., 1.));

        assert!(!t.is_intersected_by_ray(&ray));
    }

    #[test]
    fn ray_misses_p1_p2_edge() {
        let t = get_triangle();
        let ray = Ray::new(Point::new(-1., 1., -2.), Vector::new(0., 0., 1.));

        assert!(!t.is_intersected_by_ray(&ray));
    }

    #[test]
    fn ray_misses_p2_p3_edge() {
        let t = get_triangle();
        let ray = Ray::new(Point::new(0., -1., -2.), Vector::new(0., 0., 1.));

        assert!(!t.is_intersected_by_ray(&ray));
    }

    #[test]
    fn ray_strikes_triangle() {
        let t = get_triangle();
        let ray = Ray::new(Point::new(0., 0.5, -2.), Vector::new(0., 0., 1.));

        let xs = t.intersection_times(&ray);
        assert_eq!(xs.len(), 1);
        assert!(xs[0].approx_eq(&2.));
    }
}
