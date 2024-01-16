use std::f64::consts::FRAC_PI_2;

use crate::{
    primitive::matrix4::Matrix4,
    render::{
        canvas::Canvas,
        color::Color,
        material::Material,
        object::{Object, Shape},
        pattern::Pattern,
        world::World,
    },
    transformation::{rotation_x_matrix, scaling_matrix, translation_matrix, Transform},
};

use super::making_scene;

pub fn run(width: usize, height: usize) -> Canvas {
    let c1 = Color::new(0.6, 0.6, 0.6);
    let material = Material::with_pattern(Pattern::checkers(c1, Color::black(), None));

    let floor = Object::new(Shape::Plane, material.clone(), Matrix4::identity_matrix());

    let wall = Object::new(
        Shape::Plane,
        material,
        rotation_x_matrix(FRAC_PI_2)
            .translate(0., 0., 5.)
            .get_transformed(),
    );

    let small_sphere = Object::new(
        Shape::Sphere,
        Material::glass(),
        translation_matrix(-1.5, 1., -5.)
            .scale(0.5, 0.5, 0.5)
            .get_transformed(),
    );

    let mid_sphere = Object::new(
        Shape::Sphere,
        Material::glass(),
        translation_matrix(0., 1., -1.5).get_transformed(),
    );

    let mid_sphere_air_pocket = Object::new(
        Shape::Sphere,
        Material::air(),
        scaling_matrix(0.5, 0.5, 0.5)
            .translate(0., 1., -1.5)
            .get_transformed(),
    );

    let lights = making_scene::scene_lights();

    let objects = vec![floor, wall, small_sphere, mid_sphere, mid_sphere_air_pocket];

    let world = World::new(objects, lights);

    let camera = making_scene::scene_camera(width, height);

    world.render(&camera)
}
