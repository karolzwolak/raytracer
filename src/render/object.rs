use crate::{
    approx_eq::ApproxEq,
    primitive::{
        matrix::{Matrix, Transform},
        point::Point,
        tuple::Tuple,
        vector::Vector,
    },
};

use super::{material::Material, ray::Ray};

#[derive(Copy, Clone)]
pub enum Shape {
    /// Unit sphere at point zero
    Sphere,
    /// Plane extending in x and z directions, at y = 0
    Plane,
    /// Cube with sides of length 2, centered at origin
    Cube,
    /// Cylinder with radius 1, extending infinitely in y direction
    Cylinder,
}

impl Shape {
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
            Shape::Cylinder => Vector::new(object_point.x(), 0., object_point.z()).normalize(),
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
        self.transformation = self.transformation * matrix;
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

    pub fn intersection_times(&self, ray: &Ray) -> Vec<f64> {
        let object_ray = ray.transform(self.transformation_inverse().unwrap());

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
            Shape::Cylinder => {
                let a = object_ray.direction().x().powi(2) + object_ray.direction().z().powi(2);

                // ray is parallel to the y axis
                if a.approx_eq(&0.) {
                    return Vec::new();
                }

                let b = 2. * object_ray.origin().x() * object_ray.direction().x()
                    + 2. * object_ray.origin().z() * object_ray.direction().z();
                let c = object_ray.origin().x().powi(2) + object_ray.origin().z().powi(2) - 1.;

                let discriminant = b * b - 4. * a * c;

                if discriminant < 0. {
                    return Vec::new();
                }

                let delta_sqrt = discriminant.sqrt();
                vec![(-b - delta_sqrt) / (2. * a), (-b + delta_sqrt) / (2. * a)]
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
}

#[cfg(test)]
mod tests {
    use std::{f64::consts::FRAC_1_SQRT_2, f64::consts::PI};

    use super::*;
    use crate::primitive::{matrix::Matrix, vector::Vector};

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
        let cyl = Object::with_shape(Shape::Cylinder);
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
        let cyl = Object::with_shape(Shape::Cylinder);

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
        let cyl = Object::with_shape(Shape::Cylinder);

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
}
