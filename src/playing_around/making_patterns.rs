use crate::{
    primitive::{matrix::Matrix, point::Point, tuple::Tuple},
    render::{
        canvas::Canvas,
        color::Color,
        light::PointLightSource,
        material::Material,
        object::{shape::Shape, Object, PrimitiveObject},
        pattern::Pattern,
        world::World,
    },
};

use super::making_scene;

pub fn run(width: usize, height: usize) -> Canvas {
    let floor = PrimitiveObject::with_shape_material(
        Shape::Plane,
        Material::with_pattern(Pattern::ring(
            Color::new(0.15, 0.6, 0.7),
            Color::new(0.5, 0.1, 0.4),
            Some(Matrix::scaling_uniform(0.25)),
        )),
    )
    .into();

    let sphere = Object::primitive(
        Shape::Sphere,
        Material::with_pattern(Pattern::checkers(
            Color::white(),
            Color::red(),
            // None,
            Some(Matrix::scaling_uniform(0.5)),
        )),
        Matrix::translation(0., 1., 0.),
    );

    let light = PointLightSource::new(Point::new(-10.0, 10.0, -10.0), Color::white());

    let world = World::new(vec![floor, sphere], vec![light], None);
    let camera = making_scene::scene_camera(width, height);

    world.render(&camera)
}
