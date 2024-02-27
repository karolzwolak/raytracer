use crate::{
    approx_eq::{self, ApproxEq},
    primitive::{
        matrix::{Matrix, Transform},
        point::Point,
        tuple::Tuple,
        vector::Vector,
    },
};

use super::{intersection::Intersection, material::Material, ray::Ray};

#[derive(Clone)]
/// A group of objects that can be transformed simultaneously.
/// However, children added later will not be affected by previous transformations.
pub struct ObjectGroup {
    children: Vec<Object>,
}

impl ObjectGroup {
    pub fn new(children: Vec<Object>) -> Self {
        Self { children }
    }
    pub fn with_transformations(children: Vec<Object>, transformation: Matrix) -> Self {
        let mut group = Self::new(children);
        group.apply_transformation(transformation);
        group
    }
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }
    pub fn apply_transformation(&mut self, matrix: Matrix) {
        for child in self.children.iter_mut() {
            child.apply_group_transformation(matrix);
        }
    }
    pub fn add_child(&mut self, child: Object) {
        self.children.push(child);
    }
    pub fn into_shape(self) -> Shape {
        Shape::Group(self)
    }
}

impl Default for ObjectGroup {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Clone)]
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
    Group(ObjectGroup),
}

impl Shape {
    const CYLINDER_RADIUS: f64 = 1.;

    pub fn object_normal_at(&self, object_point: Point) -> Vector {
        match self {
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
            Shape::Group(_) => {
                panic!("Internal bug: this function should not be called on a group")
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

    pub fn get_group(&self) -> Option<&ObjectGroup> {
        match self {
            Shape::Group(group) => Some(group),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct Object {
    shape: Shape,
    material: Material,
    transformation: Matrix,
}

impl Object {
    pub fn new(shape: Shape, material: Material, transformation: Matrix) -> Self {
        Self {
            shape,
            material,
            transformation,
        }
    }

    pub fn group(children: Vec<Object>, transformation: Matrix) -> Self {
        Self::with_shape(Shape::Group(ObjectGroup::with_transformations(
            children,
            transformation,
        )))
    }

    pub fn with_shape(shape: Shape) -> Self {
        Self::with_transformation(shape, Matrix::identity())
    }
    pub fn with_shape_material(shape: Shape, material: Material) -> Self {
        Self::new(shape, material, Matrix::identity())
    }
    pub fn with_transformation(shape: Shape, matrix: Matrix) -> Self {
        Self::new(shape, Material::default(), matrix)
    }
    pub fn sphere(center: Point, radius: f64) -> Self {
        Self::with_transformation(
            Shape::Sphere,
            Matrix::identity()
                .scale(radius, radius, radius)
                .translate(center.x(), center.y(), center.z())
                .transformed(),
        )
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn transformation(&self) -> &Matrix {
        &self.transformation
    }
    pub fn transformation_inverse(&self) -> Option<Matrix> {
        self.transformation.inverse()
    }
    pub fn apply_transformation(&mut self, matrix: Matrix) {
        self.transformation = matrix * self.transformation;
    }
    pub fn apply_group_transformation(&mut self, matrix: Matrix) {
        match &mut self.shape {
            Shape::Group(group) => group.apply_transformation(matrix),
            _ => self.apply_transformation(matrix),
        }
    }
    pub fn normal_vector_at(&self, world_point: Point) -> Vector {
        let inverse = self.transformation_inverse().unwrap();
        let object_point = inverse * world_point;

        let object_normal = self.shape.object_normal_at(object_point);
        let world_normal = inverse.transpose() * object_normal;
        world_normal.normalize()
    }

    fn cube_axis_intersec_times(&self, origin: f64, direction: f64) -> (f64, f64) {
        assert!(matches!(self.shape, Shape::Cube));
        let tmin_numerator = -1. - origin;
        let tmax_numerator = 1. - origin;

        let tmin = tmin_numerator / direction;
        let tmax = tmax_numerator / direction;

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

    fn intersect_cyl_caps(&self, ray: &Ray, times: &mut Vec<f64>) {
        match self.shape {
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
                    times.push(tmin);
                }
                if self.check_cap_within_radius(ray, tmax, Shape::CYLINDER_RADIUS) {
                    times.push(tmax);
                }
            }
            _ => panic!("expected Shape::Cylinder"),
        }
    }
    fn intersect_cone_caps(&self, ray: &Ray, times: &mut Vec<f64>) {
        match self.shape {
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
                    times.push(tmin);
                }
                if self.check_cap_within_radius(ray, tmax, y_max.abs()) {
                    times.push(tmax);
                }
            }
            _ => panic!("expected Shape::Cone"),
        }
    }

    pub fn intersection_times(&self, world_ray: &Ray) -> Vec<f64> {
        let object_ray = world_ray.transform(self.transformation_inverse().unwrap());

        match self.shape {
            Shape::Sphere => {
                let vector_sphere_to_ray = *object_ray.origin() - Point::new(0., 0., 0.);

                let a = object_ray.direction().dot(*object_ray.direction());
                let b = 2. * object_ray.direction().dot(vector_sphere_to_ray);
                let c = vector_sphere_to_ray.dot(vector_sphere_to_ray) - 1.;

                let discriminant = b * b - 4. * a * c;
                if discriminant < 0. || a == 0. {
                    return Vec::new();
                }

                let delta_sqrt = discriminant.sqrt();
                vec![(-b - delta_sqrt) / (2. * a), (-b + delta_sqrt) / (2. * a)]
            }
            Shape::Plane => {
                let parallel = object_ray.direction().y().approx_eq(&0.);
                if parallel {
                    return Vec::new();
                }
                vec![-object_ray.origin().y() / object_ray.direction().y()]
            }
            Shape::Cube => {
                let (xtmin, xtmax) = self
                    .cube_axis_intersec_times(object_ray.origin().x(), object_ray.direction().x());
                let (ytmin, ytmax) = self
                    .cube_axis_intersec_times(object_ray.origin().y(), object_ray.direction().y());
                let (ztmin, ztmax) = self
                    .cube_axis_intersec_times(object_ray.origin().z(), object_ray.direction().z());

                let tmin = xtmin.max(ytmin).max(ztmin);
                let tmax = xtmax.min(ytmax).min(ztmax);

                if tmin > tmax {
                    return Vec::new();
                }

                vec![tmin, tmax]
            }
            Shape::Cylinder { y_min, y_max, .. } => {
                if y_min.approx_eq(&y_max) {
                    return Vec::new();
                }

                let mut res = Vec::with_capacity(2);

                self.intersect_cyl_caps(&object_ray, &mut res);

                let a = object_ray.direction().x().powi(2) + object_ray.direction().z().powi(2);

                // ray is parallel to the y axis
                if a.approx_eq(&0.) {
                    return res;
                }

                let b = 2. * object_ray.origin().x() * object_ray.direction().x()
                    + 2. * object_ray.origin().z() * object_ray.direction().z();
                let c = object_ray.origin().x().powi(2) + object_ray.origin().z().powi(2) - 1.;

                let discriminant = b * b - 4. * a * c;

                if discriminant < 0. {
                    return Vec::new();
                }

                let delta_sqrt = discriminant.sqrt();

                let t0 = (-b - delta_sqrt) / (2. * a);
                let t1 = (-b + delta_sqrt) / (2. * a);

                let y0 = object_ray.origin().y() + t0 * object_ray.direction().y();

                if y_min < y0 && y0 < y_max {
                    res.push(t0);
                }

                let y1 = object_ray.origin().y() + t1 * object_ray.direction().y();

                if y_min < y1 && y1 < y_max {
                    res.push(t1);
                }

                res
            }
            Shape::Cone { y_min, y_max, .. } => {
                if y_min.approx_eq(&y_max) {
                    return Vec::new();
                }

                let mut res = Vec::with_capacity(4);

                self.intersect_cone_caps(&object_ray, &mut res);

                let dir = object_ray.direction();
                let origin = object_ray.origin();

                let a = dir.x().powi(2) - dir.y().powi(2) + dir.z().powi(2);
                let b = 2. * (origin.x() * dir.x() - origin.y() * dir.y() + origin.z() * dir.z());
                let c = origin.x().powi(2) - origin.y().powi(2) + origin.z().powi(2);

                let ray_parallel_to_one_half = a.approx_eq(&0.);

                if ray_parallel_to_one_half {
                    if b.approx_eq(&0.) {
                        return Vec::new();
                    }
                    let t = -c / (2. * b);
                    res.push(t);
                } else {
                    let discriminant = b * b - 4. * a * c;
                    if discriminant < 0. {
                        return Vec::new();
                    }
                    let delta_sqrt = discriminant.sqrt();
                    let t0 = (-b - delta_sqrt) / (2. * a);
                    let t1 = (-b + delta_sqrt) / (2. * a);
                    let y0 = origin.y() + t0 * dir.y();

                    if y_min < y0 && y0 < y_max {
                        res.push(t0);
                    }
                    let y1 = origin.y() + t1 * dir.y();
                    if y_min < y1 && y1 < y_max {
                        res.push(t1);
                    }
                }

                res
            }
            Shape::Group(_) => {
                panic!("Internal bug: this function should not be called on a group")
            }
        }
    }

    pub fn intersect<'a>(&'a self, ray: &Ray) -> Vec<Intersection<'a>> {
        match &self.shape {
            Shape::Group(ref group) => group.children.iter().fold(Vec::new(), |mut acc, child| {
                acc.extend(child.intersect(ray));
                acc
            }),
            _ => {
                let times = self.intersection_times(ray);
                times
                    .into_iter()
                    .map(|t| Intersection::new(t, self))
                    .collect()
            }
        }
    }

    pub fn is_intersected_by_ray(&self, ray: &Ray) -> bool {
        !self.intersection_times(ray).is_empty()
    }
    pub fn material(&self) -> &Material {
        &self.material
    }

    pub fn set_material(&mut self, material: Material) {
        self.material = material;
    }

    pub fn material_mut(&mut self) -> &mut Material {
        &mut self.material
    }

    pub fn get_group(&self) -> Option<&ObjectGroup> {
        self.shape.get_group()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        f64::consts::FRAC_1_SQRT_2,
        f64::consts::{PI, SQRT_2},
    };

    use super::*;
    use crate::{
        primitive::{matrix::Matrix, vector::Vector},
        render::intersection::IntersecVec,
    };

    #[test]
    fn identiy_matrix_is_obj_default_transformation() {
        assert_eq!(
            Object::with_shape(Shape::Sphere).transformation,
            Matrix::identity()
        );
    }
    #[test]
    fn transformed_sphere() {
        // < -2; 6 >
        let obj = Object::sphere(Point::new(2., 2., 2.), 4.);

        let direction = Vector::new(0., 0., 1.);
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(2., 2., 2.), direction)));
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(2., 2., -2.), direction)));
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(4., 2., -2.), direction)));
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(6., 2., -2.), direction)));
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(-2., 2., -2.), direction)));
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(-1., 2., -2.), direction)));
        assert!(obj.is_intersected_by_ray(&Ray::new(Point::new(3., -1., -2.), direction)));

        assert!(!obj.is_intersected_by_ray(&Ray::new(Point::new(-1., -2., -2.), direction)));
        assert!(!obj.is_intersected_by_ray(&Ray::new(Point::new(3., -8., -2.), direction)));
        assert!(!obj.is_intersected_by_ray(&Ray::new(Point::new(3., -6., -2.), direction)));
    }

    #[test]
    fn intersect_scaled_sphere() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let obj = Object::with_transformation(Shape::Sphere, Matrix::scaling_uniform(2.));

        assert_eq!(obj.intersection_times(&ray), vec![3., 7.]);
    }
    #[test]
    fn intersect_translated_sphere() {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let obj = Object::with_transformation(Shape::Sphere, Matrix::translation(5., 0., 0.));

        assert_eq!(obj.intersection_times(&ray), vec![]);
    }
    #[test]
    fn ray_intersecting_plane_from_above() {
        let plane = Object::with_shape(Shape::Plane);
        let ray = Ray::new(Point::new(0., 1., 0.), Vector::new(0., -1., 0.));

        assert_eq!(plane.intersection_times(&ray), vec![1.]);
    }

    #[test]
    fn ray_intersecting_plane_from_below() {
        let plane = Object::with_shape(Shape::Plane);
        let ray = Ray::new(Point::new(0., -1., 0.), Vector::new(0., 1., 0.));

        assert_eq!(plane.intersection_times(&ray), vec![1.]);
    }

    #[test]
    fn ray_intersects_cube() {
        let cube = Object::with_shape(Shape::Cube);
        let examples = vec![
            Ray::new(Point::new(5., 0.5, 0.), Vector::new(-1., 0., 0.)),
            Ray::new(Point::new(-5., 0.5, 0.), Vector::new(1., 0., 0.)),
            Ray::new(Point::new(0.5, 5., 0.), Vector::new(0., -1., 0.)),
            Ray::new(Point::new(0.5, -5., 0.), Vector::new(0., 1., 0.)),
            Ray::new(Point::new(0.5, 0., 5.), Vector::new(0., 0., -1.)),
            Ray::new(Point::new(0.5, 0., -5.), Vector::new(0., 0., 1.)),
            Ray::new(Point::new(0., 0.5, 0.), Vector::new(0., 0., 1.)),
        ];
        let expected_times = vec![
            vec![4., 6.],
            vec![4., 6.],
            vec![4., 6.],
            vec![4., 6.],
            vec![4., 6.],
            vec![4., 6.],
            vec![-1., 1.],
        ];
        assert_eq!(examples.len(), expected_times.len());

        for (ray, expected) in examples.iter().zip(expected_times.iter()) {
            assert_eq!(cube.intersection_times(ray), *expected);
        }
    }

    #[test]
    fn ray_misses_cube() {
        let cube = Object::with_shape(Shape::Cube);

        let rays = vec![
            Ray::new(Point::new(-2., 0., 0.), Vector::new(0.2673, 0.5345, 0.8018)),
            Ray::new(Point::new(0., -2., 0.), Vector::new(0.8018, 0.2673, 0.5345)),
            Ray::new(Point::new(0., 0., -2.), Vector::new(0.5345, 0.8018, 0.2673)),
            Ray::new(Point::new(2., 0., 2.), Vector::new(0., 0., -1.)),
            Ray::new(Point::new(0., 2., 2.), Vector::new(0., -1., 0.)),
            Ray::new(Point::new(2., 2., 0.), Vector::new(-1., 0., 0.)),
        ];

        for ray in rays {
            assert!(!cube.is_intersected_by_ray(&ray));
        }
    }

    #[test]
    fn normal_on_sphere_x_axis() {
        let sphere_obj = Object::with_shape(Shape::Sphere);

        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(1., 0., 0.,)),
            Vector::new(1., 0., 0.)
        );
    }
    #[test]
    fn normal_on_sphere_y_axis() {
        let sphere_obj = Object::with_shape(Shape::Sphere);

        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(0., 1., 0.,)),
            Vector::new(0., 1., 0.)
        );
    }
    #[test]
    fn normal_on_sphere_z_axis() {
        let sphere_obj = Object::with_shape(Shape::Sphere);

        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(0., 0., 1.,)),
            Vector::new(0., 0., 1.)
        );
    }
    #[test]
    fn normal_on_sphere_at_noaxial_point() {
        let sphere_obj = Object::with_shape(Shape::Sphere);

        let frac_sqrt_3_3 = 3_f64.sqrt() / 3.;
        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(frac_sqrt_3_3, frac_sqrt_3_3, frac_sqrt_3_3)),
            Vector::new(frac_sqrt_3_3, frac_sqrt_3_3, frac_sqrt_3_3)
        );
    }
    #[test]
    fn normal_is_normalized() {
        let sphere_obj = Object::with_shape(Shape::Sphere);

        let frac_sqrt_3_3 = 3_f64.sqrt() / 3.;
        let normal =
            sphere_obj.normal_vector_at(Point::new(frac_sqrt_3_3, frac_sqrt_3_3, frac_sqrt_3_3));
        assert_eq!(normal, normal.normalize());
    }
    #[test]
    fn compute_normal_on_translated_sphere() {
        let mut sphere_obj = Object::with_shape(Shape::Sphere);
        sphere_obj.apply_transformation(Matrix::translation(0., 1., 0.));
        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(0., 1. + FRAC_1_SQRT_2, -FRAC_1_SQRT_2)),
            Vector::new(0., FRAC_1_SQRT_2, -FRAC_1_SQRT_2)
        );
    }
    #[test]
    fn compute_normal_on_transformed_sphere() {
        let mut sphere_obj = Object::with_shape(Shape::Sphere);
        sphere_obj.apply_transformation(Matrix::scaling(1., 0.5, 1.) * Matrix::rotation_z(PI / 5.));
        assert_eq!(
            sphere_obj.normal_vector_at(Point::new(0., FRAC_1_SQRT_2, -FRAC_1_SQRT_2)),
            Vector::new(0., 0.97014, -0.24254)
        );
    }

    #[test]
    fn normal_of_plane_is_const_everywhere() {
        let plane = Object::with_shape(Shape::Plane);

        let expected = Vector::new(0., 1., 0.);

        assert_eq!(plane.normal_vector_at(Point::new(0., 0., 0.,)), expected);
        assert_eq!(plane.normal_vector_at(Point::new(10., 0., -10.,)), expected);
        assert_eq!(plane.normal_vector_at(Point::new(-5., 0., 150.,)), expected);
    }

    #[test]
    fn normal_on_surface_of_cube() {
        let cube = Object::with_shape(Shape::Cube);
        let examples = vec![
            (Point::new(1., 0.5, -0.8), Vector::new(1., 0., 0.)),
            (Point::new(-1., -0.2, 0.9), Vector::new(-1., 0., 0.)),
            (Point::new(-0.4, 1., -0.1), Vector::new(0., 1., 0.)),
            (Point::new(0.3, -1., -0.7), Vector::new(0., -1., 0.)),
            (Point::new(-0.6, 0.3, 1.), Vector::new(0., 0., 1.)),
            (Point::new(0.4, 0.4, -1.), Vector::new(0., 0., -1.)),
            (Point::new(1., 1., 1.), Vector::new(1., 0., 0.)),
            (Point::new(-1., -1., -1.), Vector::new(-1., 0., 0.)),
        ];

        for (point, expected) in examples {
            assert_eq!(cube.normal_vector_at(point), expected);
        }
    }

    #[test]
    fn ray_misses_cylinder() {
        let cyl = Object::with_shape(Shape::default_cylinder());
        let examples = vec![
            Ray::new(Point::new(1., 0., 0.), Vector::new(0., 1., 0.)),
            Ray::new(Point::new(0., 0., 0.), Vector::new(0., 1., 0.)),
            Ray::new(Point::new(0., 0., -5.), Vector::new(1., 1., 1.)),
        ];

        for ray in examples {
            assert!(!cyl.is_intersected_by_ray(&ray));
        }
    }

    #[test]
    fn ray_intersects_cylinder() {
        let cyl = Object::with_shape(Shape::default_cylinder());

        let examples = vec![
            (
                Point::new(1., 0., -5.),
                Vector::new(0., 0., 1.),
                vec![5., 5.],
            ),
            (
                Point::new(0., 0., -5.),
                Vector::new(0., 0., 1.),
                vec![4., 6.],
            ),
            (
                Point::new(0.5, 0., -5.),
                Vector::new(0.1, 1., 1.),
                vec![6.80798, 7.08872],
            ),
        ];

        for (origin, direction, expected) in examples {
            let ray = Ray::new(origin, direction.normalize());
            let times = cyl.intersection_times(&ray);

            assert_eq!(times.len(), expected.len());
            for t in times.iter().zip(expected.iter()) {
                assert!(t.0.approx_eq(t.1));
            }
        }
    }

    #[test]
    fn normal_of_cylinder() {
        let cyl = Object::with_shape(Shape::default_cylinder());

        let examples = vec![
            (Point::new(1., 0., 0.), Vector::new(1., 0., 0.)),
            (Point::new(0., 5., -1.), Vector::new(0., 0., -1.)),
            (Point::new(0., -2., 1.), Vector::new(0., 0., 1.)),
            (Point::new(-1., 1., 0.), Vector::new(-1., 0., 0.)),
        ];

        for (point, expected) in examples {
            assert_eq!(cyl.normal_vector_at(point), expected);
        }
    }

    #[test]
    fn default_min_max_for_cylinder() {
        let cyl = Shape::default_cylinder();

        if let Shape::Cylinder {
            y_min,
            y_max,
            closed,
        } = cyl
        {
            assert_eq!(y_min, f64::NEG_INFINITY);
            assert_eq!(y_max, f64::INFINITY);
            assert!(!closed);
        } else {
            panic!("Expected cylinder");
        }
    }

    #[test]
    fn intersecting_constrained_cylinder() {
        let cyl = Object::with_shape(Shape::Cylinder {
            y_min: 1.,
            y_max: 2.,
            closed: false,
        });

        let examples = vec![
            (Point::new(0., 1.5, 0.), Vector::new(0.1, 1., 0.), 0),
            (Point::new(0., 3., -5.), Vector::new(0., 0., 1.), 0),
            (Point::new(0., 0., -5.), Vector::new(0., 0., 1.), 0),
            (Point::new(0., 2., -5.), Vector::new(0., 0., 1.), 0),
            (Point::new(0., 1., -5.), Vector::new(0., 0., 1.), 0),
            (Point::new(0., 1.5, -2.), Vector::new(0., 0., 1.), 2),
        ];

        for (origin, direction, expected) in examples {
            let ray = Ray::new(origin, direction.normalize());
            let times = cyl.intersection_times(&ray);
            assert_eq!(times.len(), expected);
        }
    }

    #[test]
    fn intersecting_cylinder_end_caps() {
        let cyl = Object::with_shape(Shape::Cylinder {
            y_min: 1.,
            y_max: 2.,
            closed: true,
        });

        let examples = vec![
            (Point::new(0., 3., 0.), Vector::new(0., -1., 0.), 2),
            (Point::new(0., 3., -2.), Vector::new(0., -1., 2.), 2),
            (Point::new(0., 4., -2.), Vector::new(0., -1., 1.), 2), // corner case
            (Point::new(0., 0., -2.), Vector::new(0., 1., 2.), 2),
            (Point::new(0., -1., -2.), Vector::new(0., 1., 1.), 2), // corner case
        ];

        for (origin, direction, expected) in examples {
            let ray = Ray::new(origin, direction.normalize());
            let times = cyl.intersection_times(&ray);
            assert_eq!(times.len(), expected);
        }
    }

    #[test]
    fn normal_of_cylinder_end_caps() {
        let cyl = Object::with_shape(Shape::Cylinder {
            y_min: 1.,
            y_max: 2.,
            closed: true,
        });

        let examples = vec![
            (Point::new(0., 1., 0.), Vector::new(0., -1., 0.)),
            (Point::new(0.5, 1., 0.), Vector::new(0., -1., 0.)),
            (Point::new(0., 1., 0.5), Vector::new(0., -1., 0.)),
            (Point::new(0., 2., 0.), Vector::new(0., 1., 0.)),
            (Point::new(0.5, 2., 0.), Vector::new(0., 1., 0.)),
            (Point::new(0., 2., 0.5), Vector::new(0., 1., 0.)),
        ];

        for (point, expected) in examples {
            assert_eq!(cyl.normal_vector_at(point), expected);
        }
    }

    #[test]
    fn intersecting_cone() {
        let cone = Object::with_shape(Shape::default_cone());

        let examples = vec![
            (Point::new(0., 0., -5.), Vector::new(0., 0., 1.), (5., 5.)),
            (
                Point::new(0., 0., -5.),
                Vector::new(1., 1., 1.),
                (8.66025, 8.66025),
            ),
            (
                Point::new(1., 1., -5.),
                Vector::new(-0.5, -1., 1.),
                (4.55006, 49.44994),
            ),
        ];

        for (origin, direction, expected) in examples {
            let ray = Ray::new(origin, direction.normalize());
            let times = cone.intersection_times(&ray);

            assert_eq!(times.len(), 2);
            assert!(times[0].approx_eq(&expected.0));
            assert!(times[1].approx_eq(&expected.1));
        }
    }

    #[test]
    fn intersecting_cone_with_ray_parallel_to_one_half() {
        let cone = Object::with_shape(Shape::default_cone());
        let ray = Ray::new(Point::new(0., 0., -1.), Vector::new(0., 1., 1.).normalize());
        let times = cone.intersection_times(&ray);

        assert_eq!(times.len(), 1);
        assert!(times[0].approx_eq(&0.35355));
    }

    #[test]
    fn intersecting_cone_caps() {
        let cone = Object::with_shape(Shape::Cone {
            y_min: -0.5,
            y_max: 0.5,
            closed: true,
        });
        let examples = vec![
            (Point::new(0., 0., -5.), Vector::new(0., 1., 0.), 0),
            (Point::new(0., 0., -0.25), Vector::new(0., 1., 1.), 2),
            (Point::new(0., 0., -0.25), Vector::new(0., 1., 0.), 4),
        ];

        for (origin, direction, expected) in examples {
            let ray = Ray::new(origin, direction.normalize());
            let times = cone.intersection_times(&ray);
            assert_eq!(times.len(), expected);
        }
    }

    #[test]
    fn normal_of_cone_caps() {
        let cone = Object::with_shape(Shape::default_cone());

        let examples = vec![
            (Point::new(0., 0., 0.), Vector::new(0., 0., 0.)),
            (Point::new(1., 1., 1.), Vector::new(1., -SQRT_2, 1.)),
            (Point::new(-1., -1., 0.), Vector::new(-1., 1., 0.)),
        ];

        for (point, expected) in examples {
            assert_eq!(cone.normal_vector_at(point), expected.normalize());
        }
    }

    #[test]
    fn intersecting_ray_with_empty_group() {
        let group = ObjectGroup::empty();
        let object = Object::with_shape(group.into_shape());
        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        assert!(object.intersect(&ray).is_empty());
    }

    #[test]
    fn intersecting_ray_with_nonempty_group() {
        let s1 = Object::with_shape(Shape::Sphere);
        let s2 = Object::sphere(Point::new(0., 0., -3.), 1.);
        let s3 = Object::sphere(Point::new(5., 0., 0.), 1.);

        let group = ObjectGroup::new(vec![s1, s2, s3]);
        let object = Object::with_shape(group.into_shape());

        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let xs = IntersecVec::from_ray_and_obj(ray, &object);
        let data = xs.data();

        match object.shape() {
            Shape::Group(group) => {
                assert_eq!(data.len(), 4);

                assert!(std::ptr::eq(data[0].object(), &group.children[1]));
                assert!(std::ptr::eq(data[1].object(), &group.children[1]));
                assert!(std::ptr::eq(data[2].object(), &group.children[0]));
                assert!(std::ptr::eq(data[3].object(), &group.children[0]));
            }
            _ => panic!("expected Shape::Group"),
        }
    }

    #[test]
    fn intersecting_transformed_group() {
        let sphere = Object::sphere(Point::new(5., 0., 0.), 1.);
        let group = ObjectGroup::with_transformations(vec![sphere], Matrix::scaling_uniform(2.));
        let object = Object::with_shape(group.into_shape());

        let ray = Ray::new(Point::new(10., 0., -10.), Vector::new(0., 0., 1.));
        assert_eq!(object.intersect(&ray).len(), 2);
    }

    #[test]
    fn normal_on_group_child() {
        let sphere = Object::with_transformation(Shape::Sphere, Matrix::translation(5., 0., 0.));
        let g2 = Object::group(vec![sphere], Matrix::scaling(1., 2., 3.));
        let g1 = Object::group(vec![g2], Matrix::rotation_y(std::f64::consts::FRAC_PI_2));

        let sphere = &g1.get_group().unwrap().children[0]
            .get_group()
            .unwrap()
            .children[0];
        let normal = sphere.normal_vector_at(Point::new(1.7321, 1.1547, -5.5774));
        assert!(normal.approx_eq_low_prec(&Vector::new(0.2857, 0.4286, -0.8571)));
    }
}
