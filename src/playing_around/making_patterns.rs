use crate::{
    primitive::{matrix4::Matrix4, point::Point, tuple::Tuple},
    render::{
        canvas::Canvas,
        color::Color,
        light::PointLightSource,
        material::Material,
        object::{Object, Shape},
        pattern::Pattern,
        world::World,
    },
    transformation::{scaling_matrix, translation_matrix},
};

use super::making_scene;

pub fn run() -> Canvas {
    let floor = Object::with_shape_material(
        Shape::Plane,
        Material::with_pattern(Pattern::ring(
            Color::new(0.15, 0.6, 0.7),
            Color::new(0.5, 0.1, 0.4),
            Some(scaling_matrix(0.25, 0.25, 0.25)),
        )),
    );

    let sphere = Object::new(
        Shape::Sphere,
        Material::with_pattern(Pattern::checkers(
            Color::white(),
            Color::red(),
            // None,
            Some(scaling_matrix(0.5, 0.5, 0.5)),
        )),
        translation_matrix(0., 1., 0.),
    );

    let light = PointLightSource::new(Point::new(-10.0, 10.0, -10.0), Color::white());

    let world = World::new(vec![floor, sphere], vec![light]);
    let camera = making_scene::scene_camera(800, 800);

    world.render(&camera)
}
