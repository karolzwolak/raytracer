use std::io;

use crate::render::{
    object::{Object, Shape},
    world::World,
};

use super::making_scene;

pub fn run(filename: &str) -> Result<(), io::Error> {
    let mut objects = making_scene::scene_objects();
    let lights = making_scene::scene_lights();
    objects.push(Object::with_shape(Shape::Plane));

    let world = World::new(objects, lights);
    let camera = making_scene::scene_camera();

    world.render(&camera).save_to_file(filename)
}
