use crate::{
    approx_eq::ApproxEq,
    core::{
        matrix::{Matrix, Transform},
        point::Point,
        vector::Vector,
    },
    render::{
        intersection::{Intersection, IntersectionCollector},
        ray::Ray,
    },
    BoundingBox,
};

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

impl Transform for SmoothTriangle {
    fn transform(&mut self, matrix: &Matrix) {
        *self = self.transform_new(matrix);
    }

    fn transform_new(&self, matrix: &Matrix) -> Self {
        let p1 = matrix * self.p1;
        let p2 = matrix * self.p2;
        let p3 = matrix * self.p3;
        let n1 = matrix * self.n1;
        let n2 = matrix * self.n2;
        let n3 = matrix * self.n3;

        Self::new(p1, p2, p3, n1, n2, n3)
    }
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
    pub fn local_normal_at<'a>(&self, i: Option<&'a Intersection<'a>>) -> Vector {
        let i = i.unwrap();
        self.n2 * i.u() + self.n3 * i.v() + self.n1 * (1. - i.u() - i.v())
    }
    pub fn local_intersect(&self, object_ray: &Ray, collector: &mut IntersectionCollector) {
        let dir_cross_e2 = object_ray.direction().cross(self.e2());
        let det = self.e1().dot(dir_cross_e2);

        if det.approx_eq(&0.) {
            return;
        }

        let f = 1. / det;
        let p1_to_origin = *object_ray.origin() - self.p1();
        let u = f * p1_to_origin.dot(dir_cross_e2);
        if !(0.0..=1.).contains(&u) {
            return;
        }

        let origin_cross_e1 = p1_to_origin.cross(self.e1());
        let v = f * object_ray.direction().dot(origin_cross_e1);
        if v < 0. || u + v > 1. {
            return;
        }

        let t = f * self.e2().dot(origin_cross_e1);
        collector.add_uv(t, u, v);
    }

    pub fn bounding_box(&self) -> BoundingBox {
        let mut bounding_box = BoundingBox::empty();
        bounding_box.add_point(self.p1);
        bounding_box.add_point(self.p2);
        bounding_box.add_point(self.p3);
        bounding_box
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
        assert_approx_eq_low_prec,
        core::{point::Point, tuple::Tuple, vector::Vector},
        render::{intersection::Intersection, ray::Ray},
        Object, Shape,
    };

    fn get_smooth_triangle() -> Object {
        let p1 = Point::new(0., 1., 0.);
        let p2 = Point::new(-1., 0., 0.);
        let p3 = Point::new(1., 0., 0.);
        let n1 = Vector::new(0., 1., 0.);
        let n2 = Vector::new(-1., 0., 0.);
        let n3 = Vector::new(1., 0., 0.);
        Object::core_with_shape(Shape::smooth_triangle(p1, p2, p3, n1, n2, n3))
    }

    #[test]
    fn intersection_with_smooth_triangle_store_u_v() {
        let ray = Ray::new(Point::new(-0.2, 0.3, -2.), Vector::new(0., 0., 1.));
        let triangle = get_smooth_triangle();

        let xs = triangle.intersect_to_sorted_vec_testing(&ray);
        let i = xs[0];
        assert!(i.u().approx_eq(&0.45));
        assert!(i.v().approx_eq(&0.25));
    }

    #[test]
    fn smooth_triangle_uses_uv_to_interpolate_normal() {
        let t = get_smooth_triangle();
        let i = Intersection::new_with_uv(1., &t, 0.45, 0.25);

        assert_approx_eq_low_prec!(
            t.normal_vector_at_with_intersection(Point::new(0., 0., 0.), Some(&i)),
            Vector::new(-0.5547, 0.83205, 0.)
        );
    }

    #[test]
    fn preparing_normal_on_smooth_triangle() {
        let t = get_smooth_triangle();
        let i = Intersection::new_with_uv(1., &t, 0.45, 0.25);
        let n = t.normal_vector_at_with_intersection(Point::new(0., 0., 0.), Some(&i));

        assert_approx_eq_low_prec!(n, Vector::new(-0.5547, 0.83205, 0.));
    }
}
