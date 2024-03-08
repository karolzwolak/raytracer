use crate::{
    approx_eq::{self, ApproxEq},
    primitive::{point::Point, tuple::Tuple, vector::Vector},
    render::{
        intersection::{Intersection, IntersectionCollector},
        object::triangle::Triangle,
        ray::Ray,
    },
};

use super::{bounding_box::BoundingBox, group::ObjectGroup, smooth_triangle::SmoothTriangle};

#[derive(Clone, Debug)]
pub enum Shape {
    /// Unit sphere at point zero
    Sphere,
    /// Plane extending in x and z directions, at y = 0
    Plane,
    /// Cube with sides of length 2, centered at origin
    Cube,
    /// Cylinder with radius 1, extending from y_min to y_max exclusively
    Cylinder {
        y_min: f64,
        y_max: f64,
        closed: bool,
    },
    /// Double-sided cone, their tips meeting at the origin, extending from y_min to y_max exclusively
    Cone {
        y_min: f64,
        y_max: f64,
        closed: bool,
    },
    Triangle(Triangle),
    SmoothTriangle(SmoothTriangle),
    Group(ObjectGroup),
}

impl Shape {
    const CYLINDER_RADIUS: f64 = 1.;

    pub fn local_normal_at<'a>(
        &self,
        object_point: Point,
        i: Option<&'a Intersection<'a>>,
    ) -> Vector {
        match *self {
            Shape::Sphere => object_point - Point::zero(),
            Shape::Plane => Vector::new(0., 1., 0.),
            Shape::Cube => {
                let abs_x = object_point.x().abs();
                let abs_y = object_point.y().abs();
                let abs_z = object_point.z().abs();
                let max = abs_x.max(abs_y).max(abs_z);

                if max == abs_x {
                    Vector::new(object_point.x(), 0., 0.)
                } else if max == abs_y {
                    Vector::new(0., object_point.y(), 0.)
                } else {
                    Vector::new(0., 0., object_point.z())
                }
            }
            Shape::Cylinder { y_min, y_max, .. } => {
                let dist = object_point.x().powi(2) + object_point.z().powi(2);
                let point_within_radius = dist < Self::CYLINDER_RADIUS;

                if point_within_radius && object_point.y() >= y_max - approx_eq::EPSILON {
                    Vector::new(0., 1., 0.)
                } else if point_within_radius && object_point.y() <= y_min + approx_eq::EPSILON {
                    Vector::new(0., -1., 0.)
                } else {
                    Vector::new(object_point.x(), 0., object_point.z())
                }
            }
            Shape::Cone { y_min, y_max, .. } => {
                let dist = object_point.x().powi(2) + object_point.z().powi(2);

                let min_radius = y_min.abs();
                let max_radius = y_max.abs();

                if dist < max_radius && object_point.y() >= y_max - approx_eq::EPSILON {
                    Vector::new(0., 1., 0.)
                } else if dist < min_radius && object_point.y() <= y_min + approx_eq::EPSILON {
                    Vector::new(0., -1., 0.)
                } else {
                    let mut y = (object_point.x().powi(2) + object_point.z().powi(2)).sqrt();
                    if object_point.y() > 0. {
                        y = -y;
                    }
                    Vector::new(object_point.x(), y, object_point.z())
                }
            }
            Shape::Triangle(ref triangle) => triangle.normal(),
            Shape::SmoothTriangle(ref triangle) => {
                let i = i.unwrap();
                triangle.n2() * i.u() + triangle.n3() * i.v() + triangle.n1() * (1. - i.u() - i.v())
            }
            Shape::Group(_) => {
                panic!("Internal bug: this function should not be called on a group")
            }
        }
    }
    pub fn bounding_box(&self) -> BoundingBox {
        match self {
            Shape::Sphere | Shape::Cube => BoundingBox {
                min: Point::new(-1., -1., -1.),
                max: Point::new(1., 1., 1.),
            },
            Shape::Plane => BoundingBox {
                min: Point::new(f64::NEG_INFINITY, 0., f64::NEG_INFINITY),
                max: Point::new(f64::INFINITY, 0., f64::INFINITY),
            },
            Shape::Cylinder {
                ref y_min, y_max, ..
            } => BoundingBox {
                min: Point::new(-1., *y_min, -1.),
                max: Point::new(1., *y_max, 1.),
            },
            Shape::Cone { y_min, y_max, .. } => BoundingBox {
                min: Point::new(*y_min, *y_min, *y_min),
                max: Point::new(*y_max, *y_max, *y_max),
            },
            Shape::Triangle(ref triangle) => {
                let mut bounding_box = BoundingBox::empty();
                bounding_box.add_point(triangle.p1());
                bounding_box.add_point(triangle.p2());
                bounding_box.add_point(triangle.p3());
                bounding_box
            }
            Shape::SmoothTriangle(ref triangle) => {
                let mut bounding_box = BoundingBox::empty();
                bounding_box.add_point(triangle.p1());
                bounding_box.add_point(triangle.p2());
                bounding_box.add_point(triangle.p3());
                bounding_box
            }
            Shape::Group(ref group) => {
                group
                    .children()
                    .iter()
                    .fold(BoundingBox::empty(), |acc, child| {
                        let mut new_bounding_box = child.bounding_box();
                        new_bounding_box.add_bounding_box(acc);
                        new_bounding_box
                    })
            }
        }
    }
    pub fn cylinder(height: f64, closed: bool) -> Self {
        assert!(height >= 0.);

        let (y_min, y_max) = if height.approx_eq(&0.) {
            (f64::NEG_INFINITY, f64::INFINITY)
        } else {
            (-height / 2., height / 2.)
        };

        Shape::Cylinder {
            y_min,
            y_max,
            closed,
        }
    }

    pub fn default_cylinder() -> Self {
        Shape::cylinder(0., false)
    }

    pub fn unit_cylinder() -> Self {
        Shape::cylinder(1., true)
    }

    pub fn cone(height: f64, y_offset: f64, closed: bool) -> Self {
        assert!(height >= 0.);

        let (y_min, y_max) = if height.approx_eq(&0.) {
            (f64::NEG_INFINITY, f64::INFINITY)
        } else {
            (-height / 2. + y_offset, height / 2. + y_offset)
        };

        Shape::Cone {
            y_min,
            y_max,
            closed,
        }
    }

    pub fn default_cone() -> Self {
        Shape::Cone {
            y_min: f64::NEG_INFINITY,
            y_max: f64::INFINITY,
            closed: false,
        }
    }

    pub fn unit_cone() -> Self {
        Shape::cone(1., -0.5, true)
    }

    pub fn triangle(p1: Point, p2: Point, p3: Point) -> Self {
        Shape::Triangle(Triangle::new(p1, p2, p3))
    }

    pub fn smooth_triangle(
        p1: Point,
        p2: Point,
        p3: Point,
        n1: Vector,
        n2: Vector,
        n3: Vector,
    ) -> Self {
        Shape::SmoothTriangle(SmoothTriangle::new(p1, p2, p3, n1, n2, n3))
    }

    pub fn get_group(&self) -> Option<&ObjectGroup> {
        match self {
            Shape::Group(group) => Some(group),
            _ => None,
        }
    }

    fn cube_axis_intersec_times(&self, origin: f64, dir_inv: f64) -> (f64, f64) {
        assert!(matches!(self, Shape::Cube));
        let tmin_numerator = -1. - origin;
        let tmax_numerator = 1. - origin;

        let tmin = tmin_numerator * dir_inv;
        let tmax = tmax_numerator * dir_inv;

        if tmin < tmax {
            (tmin, tmax)
        } else {
            (tmax, tmin)
        }
    }

    fn check_cap_within_radius(&self, ray: &Ray, t: f64, radius: f64) -> bool {
        assert!(radius >= 0.);
        let x = ray.origin().x() + t * ray.direction().x();
        let z = ray.origin().z() + t * ray.direction().z();

        x * x + z * z <= radius
    }

    fn intersect_cyl_caps(&self, ray: &Ray, collector: &mut IntersectionCollector) {
        match self {
            Shape::Cylinder {
                y_min,
                y_max,
                closed,
            } => {
                if !closed || ray.direction().y().approx_eq(&0.) {
                    return;
                }
                let tmin = (y_min - ray.origin().y()) / ray.direction().y();
                let tmax = (y_max - ray.origin().y()) / ray.direction().y();

                if self.check_cap_within_radius(ray, tmin, Shape::CYLINDER_RADIUS) {
                    collector.add(tmin);
                }
                if self.check_cap_within_radius(ray, tmax, Shape::CYLINDER_RADIUS) {
                    collector.add(tmax);
                }
            }
            _ => panic!("expected Shape::Cylinder"),
        }
    }
    fn intersect_cone_caps(&self, ray: &Ray, collector: &mut IntersectionCollector) {
        match self {
            Shape::Cone {
                y_min,
                y_max,
                closed,
            } => {
                if !closed || ray.direction().y().approx_eq(&0.) {
                    return;
                }
                let tmin = (y_min - ray.origin().y()) / ray.direction().y();
                let tmax = (y_max - ray.origin().y()) / ray.direction().y();

                if self.check_cap_within_radius(ray, tmin, y_min.abs()) {
                    collector.add(tmin);
                }
                if self.check_cap_within_radius(ray, tmax, y_max.abs()) {
                    collector.add(tmax);
                }
            }
            _ => panic!("expected Shape::Cone"),
        }
    }

    pub fn local_intersect(&self, object_ray: &Ray, collector: &mut IntersectionCollector) {
        match *self {
            Shape::Sphere => {
                let vector_sphere_to_ray = *object_ray.origin() - Point::new(0., 0., 0.);

                let a = object_ray.direction().dot(*object_ray.direction());
                let b = 2. * object_ray.direction().dot(vector_sphere_to_ray);
                let c = vector_sphere_to_ray.dot(vector_sphere_to_ray) - 1.;

                let discriminant = b * b - 4. * a * c;
                if discriminant < 0. || a == 0. {
                    return;
                }

                let delta_sqrt = discriminant.sqrt();
                collector.add((-b - delta_sqrt) / (2. * a));
                collector.add((-b + delta_sqrt) / (2. * a));
            }
            Shape::Plane => {
                let parallel = object_ray.direction().y().approx_eq(&0.);
                if parallel {
                    return;
                }
                collector.add(-object_ray.origin().y() / object_ray.direction().y());
            }
            Shape::Cube => {
                let (xtmin, xtmax) = self
                    .cube_axis_intersec_times(object_ray.origin().x(), object_ray.dir_inv().x());
                let (ytmin, ytmax) = self
                    .cube_axis_intersec_times(object_ray.origin().y(), object_ray.dir_inv().y());
                let (ztmin, ztmax) = self
                    .cube_axis_intersec_times(object_ray.origin().z(), object_ray.dir_inv().z());

                let tmin = xtmin.max(ytmin).max(ztmin);
                let tmax = xtmax.min(ytmax).min(ztmax);

                if tmin > tmax {
                    return;
                }

                collector.add(tmin);
                collector.add(tmax);
            }
            Shape::Cylinder { y_min, y_max, .. } => {
                if y_min.approx_eq(&y_max) {
                    return;
                }

                self.intersect_cyl_caps(object_ray, collector);

                let a = object_ray.direction().x().powi(2) + object_ray.direction().z().powi(2);

                // ray is parallel to the y axis
                if a.approx_eq(&0.) {
                    return;
                }

                let b = 2. * object_ray.origin().x() * object_ray.direction().x()
                    + 2. * object_ray.origin().z() * object_ray.direction().z();
                let c = object_ray.origin().x().powi(2) + object_ray.origin().z().powi(2) - 1.;

                let discriminant = b * b - 4. * a * c;

                if discriminant < 0. {
                    return;
                }

                let delta_sqrt = discriminant.sqrt();

                let t0 = (-b - delta_sqrt) / (2. * a);
                let t1 = (-b + delta_sqrt) / (2. * a);

                let y0 = object_ray.origin().y() + t0 * object_ray.direction().y();

                if y_min < y0 && y0 < y_max {
                    collector.add(t0);
                }

                let y1 = object_ray.origin().y() + t1 * object_ray.direction().y();

                if y_min < y1 && y1 < y_max {
                    collector.add(t1);
                }
            }
            Shape::Cone { y_min, y_max, .. } => {
                if y_min.approx_eq(&y_max) {
                    return;
                }
                self.intersect_cone_caps(object_ray, collector);

                let dir = object_ray.direction();
                let origin = object_ray.origin();

                let a = dir.x().powi(2) - dir.y().powi(2) + dir.z().powi(2);
                let b = 2. * (origin.x() * dir.x() - origin.y() * dir.y() + origin.z() * dir.z());
                let c = origin.x().powi(2) - origin.y().powi(2) + origin.z().powi(2);

                let ray_parallel_to_one_half = a.approx_eq(&0.);

                if ray_parallel_to_one_half {
                    if b.approx_eq(&0.) {
                        return;
                    }
                    let t = -c / (2. * b);
                    collector.add(t);
                } else {
                    let discriminant = b * b - 4. * a * c;
                    if discriminant < 0. {
                        return;
                    }
                    let delta_sqrt = discriminant.sqrt();
                    let t0 = (-b - delta_sqrt) / (2. * a);
                    let t1 = (-b + delta_sqrt) / (2. * a);
                    let y0 = origin.y() + t0 * dir.y();

                    if y_min < y0 && y0 < y_max {
                        collector.add(t0);
                    }
                    let y1 = origin.y() + t1 * dir.y();
                    if y_min < y1 && y1 < y_max {
                        collector.add(t1);
                    }
                }
            }
            Shape::Triangle(ref triangle) => {
                let dir_cross_e2 = object_ray.direction().cross(triangle.e2());
                let det = triangle.e1().dot(dir_cross_e2);

                if det.approx_eq(&0.) {
                    return;
                }

                let f = 1. / det;
                let p1_to_origin = *object_ray.origin() - triangle.p1();
                let u = f * p1_to_origin.dot(dir_cross_e2);
                if !(0.0..=1.).contains(&u) {
                    return;
                }

                let origin_cross_e1 = p1_to_origin.cross(triangle.e1());
                let v = f * object_ray.direction().dot(origin_cross_e1);
                if v < 0. || u + v > 1. {
                    return;
                }

                let t = f * triangle.e2().dot(origin_cross_e1);
                collector.add(t);
            }
            Shape::SmoothTriangle(ref triangle) => {
                let dir_cross_e2 = object_ray.direction().cross(triangle.e2());
                let det = triangle.e1().dot(dir_cross_e2);

                if det.approx_eq(&0.) {
                    return;
                }

                let f = 1. / det;
                let p1_to_origin = *object_ray.origin() - triangle.p1();
                let u = f * p1_to_origin.dot(dir_cross_e2);
                if !(0.0..=1.).contains(&u) {
                    return;
                }

                let origin_cross_e1 = p1_to_origin.cross(triangle.e1());
                let v = f * object_ray.direction().dot(origin_cross_e1);
                if v < 0. || u + v > 1. {
                    return;
                }

                let t = f * triangle.e2().dot(origin_cross_e1);
                collector.add_uv(t, u, v);
            }
            Shape::Group(_) => {
                panic!("Internal bug: this function should not be called on a group")
            }
        }
    }
}
