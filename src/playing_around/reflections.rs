use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

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
    transformation::{rotation_x_matrix, translation_matrix, Transform},
};

use super::making_scene;

pub fn get_walls() -> Vec<Object> {
    let material = Material::with_pattern(Pattern::checkers(
        Color::new(0.6, 0.6, 0.6),
        Color::black(),
        None,
    ));

    let dist = 12.;

    let mut floor = Object::new(Shape::Plane, material.clone(), Matrix4::identity_matrix());
    floor.material_mut().reflectivity = 0.4;

    let left_wall = Object::new(
        Shape::Plane,
        material.clone(),
        rotation_x_matrix(FRAC_PI_2)
            .rotate_y(-FRAC_PI_4)
            .translate(0., 0., dist)
            .get_transformed(),
    );

    let right_wall = Object::new(
        Shape::Plane,
        material,
        rotation_x_matrix(FRAC_PI_2)
            .rotate_y(FRAC_PI_4)
            .translate(0., 0., dist)
            .get_transformed(),
    );

    let mut l_wall_mirror = left_wall.clone();
    l_wall_mirror.apply_transformation(translation_matrix(0., -2. * dist, 0.));

    let mut r_wall_mirror = right_wall.clone();
    r_wall_mirror.apply_transformation(translation_matrix(0., -2. * dist, 0.));

    vec![floor, left_wall, right_wall, l_wall_mirror, r_wall_mirror]
}

pub fn run(width: usize, height: usize) -> Canvas {
    let mut objects = making_scene::scene_objects();

    let mirror = Material {
        pattern: Pattern::Const(Color::black()),
        reflectivity: 1.,
        ..Default::default()
    };

    let lights = making_scene::scene_lights();

    objects.get_mut(1).unwrap().set_material(mirror.clone());

    objects.get_mut(0).unwrap().set_material(mirror.clone());

    let mirror_dist = 3.;

    let mirror_wall = Object::new(
        Shape::Plane,
        mirror.clone(),
        rotation_x_matrix(FRAC_PI_2)
            .rotate_y(FRAC_PI_4)
            .translate(0., 0., mirror_dist)
            .get_transformed(),
    );

    let mirror_wall2 = Object::new(
        Shape::Plane,
        mirror,
        rotation_x_matrix(FRAC_PI_2)
            .rotate_y(-FRAC_PI_4)
            .translate(0., 0., mirror_dist)
            .get_transformed(),
    );

    objects.extend(get_walls());
    objects.push(mirror_wall);
    objects.push(mirror_wall2);

    let world = World::new(objects, lights, Some(8));
    let camera = making_scene::scene_camera(width, height);

    world.render(&camera)
}