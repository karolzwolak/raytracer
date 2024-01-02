use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4};

use crate::{
    primitive::{point::Point, tuple::Tuple, vector::Vector},
    render::{
        camera::Camera, canvas::Canvas, color::Color, light::PointLightSource, material::Material,
        object::Object, object::Shape, pattern::Pattern, world::World,
    },
    transformation::{scaling_matrix, translation_matrix, view_tranformation_matrix, Transform},
};

pub fn scene_objects() -> Vec<Object> {
    let mut middle_sphere =
        Object::with_transformation(Shape::Sphere, translation_matrix(-0.5, 1., 0.5));

    middle_sphere.set_material(Material {
        pattern: Pattern::Const(Color::new(0.1, 1., 0.5)),
        diffuse: 0.7,
        specular: 0.3,
        ..Default::default()
    });

    let mut right_sphere = Object::new(
        Shape::Sphere,
        middle_sphere.material().clone(),
        scaling_matrix(0.5, 0.5, 0.5)
            .translate(1.5, 0.5, -0.5)
            .get_transformed(),
    );
    right_sphere.material_mut().pattern = Pattern::Const(Color::new(0.5, 1., 0.1));

    let mut left_sphere = Object::with_transformation(
        Shape::Sphere,
        scaling_matrix(0.33, 0.33, 0.33)
            .translate(-1.5, 0.33, -0.75)
            .get_transformed(),
    );

    left_sphere.set_material(Material {
        pattern: Pattern::Const(Color::new(1., 0.8, 0.1)),
        diffuse: 0.7,
        specular: 0.3,
        ..Default::default()
    });

    vec![middle_sphere, right_sphere, left_sphere]
}

pub fn scene_walls() -> Vec<Object> {
    let mut floor = Object::with_transformation(Shape::Sphere, scaling_matrix(10., 0.01, 10.));
    floor.material_mut().specular = 0.;
    floor.material_mut().pattern = Pattern::Const(Color::new(1., 0.9, 0.9));

    let left_wall = Object::new(
        Shape::Sphere,
        floor.material().clone(),
        scaling_matrix(10., 0.01, 10.)
            .rotate_x(FRAC_PI_2)
            .rotate_y(-FRAC_PI_4)
            .translate(0., 0., 5.)
            .get_transformed(),
    );

    let right_wall = Object::new(
        Shape::Sphere,
        floor.material().clone(),
        scaling_matrix(10., 0.01, 10.)
            .rotate_x(FRAC_PI_2)
            .rotate_y(FRAC_PI_4)
            .translate(0., 0., 5.)
            .get_transformed(),
    );

    vec![floor, left_wall, right_wall]
}

pub fn scene_lights() -> Vec<PointLightSource> {
    vec![PointLightSource::new(
        Point::new(-10., 10., -10.),
        Color::white(),
    )]
}

pub fn scene_camera(width: usize, height: usize) -> Camera {
    let from = Point::new(0., 1.5, -5.);
    let to = Point::new(0., 1., 0.);
    let up_v = Vector::new(0., 1., 0.);

    Camera::with_transformation(
        width,
        height,
        FRAC_PI_3,
        view_tranformation_matrix(from, to, up_v),
    )
}

pub fn run(width: usize, height: usize) -> Canvas {
    let mut objects = scene_walls();
    objects.extend(scene_objects());

    let world = World::new(objects, scene_lights());
    let camera = scene_camera(width, height);

    world.render(&camera)
}
