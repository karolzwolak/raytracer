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
    let c1 = Color::new(0.21, 0.42, 0.35);
    let c2 = Color::new(0.82, 0.72, 0.61);

    let glass_color = Color::new(0.08, 0.2, 0.5);

    let material = Material::with_pattern(Pattern::checkers(
        c1,
        c2,
        Some(
            translation_matrix(0.5, 0.5, 0.5)
                .rotate_y(FRAC_PI_2 / 2.)
                .scale(0.5, 0.5, 0.5)
                .get_transformed(),
        ),
    ));

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
        Material {
            pattern: Pattern::Const(glass_color),
            ..Material::glass()
        },
        translation_matrix(-1., 1., -5.5)
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
        scaling_matrix(0.6, 0.6, 0.6)
            .translate(0., 1., -1.5)
            .get_transformed(),
    );

    let lights = making_scene::scene_lights();

    let objects = vec![floor, wall, small_sphere, mid_sphere, mid_sphere_air_pocket];

    let world = World::new(objects, lights, Some(8));

    let camera = making_scene::scene_camera(width, height);

    world.render(&camera)
}