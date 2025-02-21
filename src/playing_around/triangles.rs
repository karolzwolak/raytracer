use std::{f64::consts, fs};

use crate::{
    primitive::{
        matrix::{Matrix, Transform},
        point::Point,
        tuple::Tuple,
    },
    render::{
        camera::Camera,
        canvas::Canvas,
        color::Color,
        light::PointLightSource,
        material::Material,
        obj_parser::ObjParser,
        object::{shape::Shape, Object},
        pattern::Pattern,
        world::World,
    },
};

const FILENAME: &str = "samples/dragon.obj";

pub fn run(width: usize, height: usize) -> Canvas {
    let light = PointLightSource::new(Point::new(1., 3., 1.), Color::new(1., 1., 1.));

    let source = fs::read_to_string(FILENAME).unwrap();
    let mut model = ObjParser::parse_to_object(source).unwrap();
    model.normalize_to_longest_dim();
    model.center_above_oy();
    model.translate(0., -1., -3.);

    let background = Object::primitive(
        Shape::Cube,
        Material::with_pattern(Pattern::checkers(
            Color::with_uniform_intensity(0.62),
            Color::with_uniform_intensity(0.7),
            Some(Matrix::scaling_uniform(0.125)),
        )),
        Matrix::scaling_uniform(5.)
            .translate(0., 4., -4.)
            .transformed(),
    );
    let objects = vec![model, background];

    let mut world = World::new(objects, vec![light], None);

    let camera = Camera::new(width, height, consts::FRAC_PI_3);

    world.render(&camera)
}
