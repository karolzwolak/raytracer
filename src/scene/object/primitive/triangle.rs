use crate::{
    math::{
        approx_eq::ApproxEq,
        matrix::{Matrix, Transform},
        point::Point,
        vector::Vector,
    },
    render::ray::{intersection::IntersectionCollector, Ray},
    scene::object::bounding_box::BoundingBox,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Triangle {
    p1: Point,
    p2: Point,
    p3: Point,
    e1: Vector,
    e2: Vector,
    normal: Vector,
}

impl Transform for Triangle {
    fn transform(&mut self, matrix: &Matrix) {
        *self = self.transform_new(matrix);
    }

    fn transform_new(&self, matrix: &Matrix) -> Self {
        let p1 = matrix * self.p1;
        let p2 = matrix * self.p2;
        let p3 = matrix * self.p3;

        Self::new(p1, p2, p3)
    }
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
        collector.add(t);
    }

    pub fn bounding_box(&self) -> BoundingBox {
        let mut bounding_box = BoundingBox::empty();
        bounding_box.add_point(self.p1);
        bounding_box.add_point(self.p2);
        bounding_box.add_point(self.p3);
        bounding_box
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
    use crate::{
        assert_approx_eq_low_prec,
        math::{approx_eq::ApproxEq, point::Point, tuple::Tuple, vector::Vector},
        render::ray::Ray,
        scene::object::{
            primitive::{shape::Shape, triangle::Triangle},
            Object,
        },
    };

    #[test]
    fn constructing_triangle() {
        let p1 = Point::new(0., 1., 0.);
        let p2 = Point::new(-1., 0., 0.);
        let p3 = Point::new(1., 0., 0.);

        let t = Triangle::new(p1, p2, p3);

        assert_approx_eq_low_prec!(t.e1, Vector::new(-1., -1., 0.));
        assert_approx_eq_low_prec!(t.e2, Vector::new(1., -1., 0.));
        assert_approx_eq_low_prec!(t.normal, Vector::new(0., 0., -1.));
    }

    fn get_triangle() -> Object {
        Object::primitive_with_shape(Shape::triangle(
            Point::new(0., 1., 0.),
            Point::new(-1., 0., 0.),
            Point::new(1., 0., 0.),
        ))
    }

    #[test]
    fn finding_normal_on_triangle() {
        let t = get_triangle();
        let t_shape = match t.as_primitive().unwrap().shape() {
            Shape::Triangle(ref triangle) => triangle.clone(),
            _ => unreachable!(),
        };

        assert_approx_eq_low_prec!(t.normal_vector_at(Point::new(1., 0.5, 0.)), t_shape.normal);
        assert_approx_eq_low_prec!(
            t.normal_vector_at(Point::new(-0.5, 0.75, 0.)),
            t_shape.normal
        );
        assert_approx_eq_low_prec!(
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

        let xs = t.intersection_times_testing(&ray);
        assert_eq!(xs.len(), 1);
        assert!(xs[0].approx_eq(&2.));
    }
}
