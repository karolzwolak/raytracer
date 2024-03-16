use crate::{
    primitive::{point::Point, vector::Vector},
    render::{
        intersection::{Intersection, IntersectionCollector},
        object::triangle::Triangle,
        ray::Ray,
    },
};

use super::{
    bounding_box::BoundingBox, cone::Cone, cube::UnitCube, cylinder::Cylinder, plane::PlaneXZ,
    smooth_triangle::SmoothTriangle, sphere::UnitSphere,
};

#[derive(Clone, Debug)]
pub enum Shape {
    /// Unit sphere at point zero
    Sphere,
    /// Plane extending in x and z directions, at y = 0
    Plane,
    /// Cube with sides of length 2, centered at origin
    Cube,
    /// Cylinder with radius 1, extending from y_min to y_max exclusively
    Cylinder(Cylinder),
    /// Double-sided cone, their tips meeting at the origin, extending from y_min to y_max exclusively
    Cone(Cone),
    Triangle(Triangle),
    SmoothTriangle(SmoothTriangle),
}

impl Shape {
    pub fn local_normal_at<'a>(
        &self,
        object_point: Point,
        i: Option<&'a Intersection<'a>>,
    ) -> Vector {
        match self {
            Shape::Sphere => UnitSphere::local_normal_at(object_point),
            Shape::Plane => PlaneXZ::local_normal_at(),
            Shape::Cube => UnitCube::local_normal_at(object_point),
            Shape::Cylinder(cyl) => cyl.local_normal_at(object_point),
            Shape::Cone(cone) => cone.local_normal_at(object_point),
            Shape::Triangle(triangle) => triangle.normal(),
            Shape::SmoothTriangle(triangle) => triangle.local_normal_at(i),
        }
    }
    pub fn local_intersect(&self, object_ray: &Ray, collector: &mut IntersectionCollector) {
        match self {
            Self::Sphere => UnitSphere::local_intersect(object_ray, collector),
            Shape::Plane => PlaneXZ::local_intersect(object_ray, collector),
            Shape::Cube => UnitCube::local_intersect(object_ray, collector),
            Shape::Cylinder(cyl) => cyl.local_intersect(object_ray, collector),
            Shape::Cone(cone) => cone.local_intersect(object_ray, collector),
            Shape::Triangle(triangle) => triangle.local_intersect(object_ray, collector),
            Shape::SmoothTriangle(triangle) => triangle.local_intersect(object_ray, collector),
        }
    }
    pub fn bounding_box(&self) -> BoundingBox {
        match self {
            Shape::Sphere => UnitSphere::bounding_box(),
            Shape::Plane => PlaneXZ::bounding_box(),
            Shape::Cube => UnitCube::bounding_box(),
            Shape::Cylinder(cyl) => cyl.bounding_box(),
            Shape::Cone(cone) => cone.bounding_box(),
            Shape::Triangle(triangle) => triangle.bounding_box(),
            Shape::SmoothTriangle(triangle) => triangle.bounding_box(),
        }
    }
    pub fn cylinder(height: f64, closed: bool) -> Self {
        Shape::Cylinder(Cylinder::with_height(height, closed))
    }

    pub fn default_cylinder() -> Self {
        Shape::cylinder(0., false)
    }

    pub fn unit_cylinder() -> Self {
        Shape::cylinder(1., true)
    }

    pub fn cone(height: f64, y_offset: f64, closed: bool) -> Self {
        Shape::Cone(Cone::new(height, y_offset, closed))
    }

    pub fn default_cone() -> Self {
        Shape::Cone(Cone::default())
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
}
