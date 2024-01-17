use std::f64::consts;

use crate::{
    primitive::matrix::{Matrix, Transform},
    render::{
        canvas::Canvas,
        object::{Object, Shape},
        world::World,
    },
};

use super::making_scene;

pub fn run(width: usize, height: usize) -> Canvas {
    let mut objects = making_scene::scene_objects();
    let lights = making_scene::scene_lights();
    let plane = Object::with_transformation(Shape::Plane, Matrix::translation(0., 0.5, 0.));
    let plane2 = Object::with_transformation(
        Shape::Plane,
        Matrix::rotation_x(consts::FRAC_PI_2)
            .translate(0., 0., 0.)
            .transformed(),
    );
    objects.push(plane);
    objects.push(plane2);

    let world = World::new(objects, lights, None);
    let camera = making_scene::scene_camera(width, height);

    world.render(&camera)
}
