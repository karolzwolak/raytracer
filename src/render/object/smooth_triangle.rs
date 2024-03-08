use crate::primitive::{point::Point, vector::Vector};

#[derive(Clone, Debug, PartialEq)]
pub struct SmoothTriangle {
    p1: Point,
    p2: Point,
    p3: Point,
    e1: Vector,
    e2: Vector,
    n1: Vector,
    n2: Vector,
    n3: Vector,
}

impl SmoothTriangle {
    pub fn new(p1: Point, p2: Point, p3: Point, n1: Vector, n2: Vector, n3: Vector) -> Self {
        let v12 = p2 - p1;
        let v13 = p3 - p1;
        Self {
            p1,
            p2,
            p3,
            e1: v12,
            e2: v13,
            n1,
            n2,
            n3,
        }
    }

    pub fn n1(&self) -> Vector {
        self.n1
    }

    pub fn n2(&self) -> Vector {
        self.n2
    }

    pub fn n3(&self) -> Vector {
        self.n3
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
}

#[cfg(test)]
mod tests {
    use crate::{
        approx_eq::ApproxEq,
        primitive::{point::Point, tuple::Tuple, vector::Vector},
        render::{
            intersection::Intersection,
            object::{shape::Shape, Object},
            ray::Ray,
        },
    };

    fn get_smooth_triangle() -> Object {
        let p1 = Point::new(0., 1., 0.);
        let p2 = Point::new(-1., 0., 0.);
        let p3 = Point::new(1., 0., 0.);
        let n1 = Vector::new(0., 1., 0.);
        let n2 = Vector::new(-1., 0., 0.);
        let n3 = Vector::new(1., 0., 0.);
        Object::with_shape(Shape::smooth_triangle(p1, p2, p3, n1, n2, n3))
    }

    #[test]
    fn intersection_with_smooth_triangle_store_u_v() {
        let ray = Ray::new(Point::new(-0.2, 0.3, -2.), Vector::new(0., 0., 1.));
        let triangle = get_smooth_triangle();

        let xs = triangle.intersect_to_vec(&ray);
        let i = xs[0];
        assert!(i.u().approx_eq(&0.45));
        assert!(i.v().approx_eq(&0.25));
    }

    #[test]
    fn smooth_triangle_uses_uv_to_interpolate_normal() {
        let t = get_smooth_triangle();
        let i = Intersection::new_with_uv(1., &t, 0.45, 0.25);

        assert_eq!(
            t.normal_vector_at_with_intersection(Point::new(0., 0., 0.), Some(&i)),
            Vector::new(-0.5547, 0.83205, 0.)
        );
    }

    #[test]
    fn preparing_normal_on_smooth_triangle() {
        let t = get_smooth_triangle();
        let i = Intersection::new_with_uv(1., &t, 0.45, 0.25);
        let n = t.normal_vector_at_with_intersection(Point::new(0., 0., 0.), Some(&i));

        assert_eq!(n, Vector::new(-0.5547, 0.83205, 0.));
    }
}
