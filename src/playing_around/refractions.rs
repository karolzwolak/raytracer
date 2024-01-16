use crate::{
    render::{
        canvas::Canvas,
        material::Material,
        object::{Object, Shape},
        world::World,
    },
    transformation::translation_matrix,
};

use super::making_scene;

pub fn run(width: usize, height: usize) -> Canvas {
    let mut walls = making_scene::scene_walls();
    let sphere = Object::new(
        Shape::Sphere,
        Material::glass(),
        translation_matrix(0.0, 1.0, 0.0),
    );
    let lights = making_scene::scene_lights();

    walls.push(sphere);

    let world = World::new(walls, lights);

    let camera = making_scene::scene_camera(width, height);

    world.render(&camera)
}
