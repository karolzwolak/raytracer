use std::{f64::consts, fs};

use crate::{
    primitive::{
        matrix::{Matrix, Transform},
        point::Point,
        tuple::Tuple,
    },
    render::{
        camera::Camera, canvas::Canvas, color::Color, light::PointLightSource,
        obj_parser::ObjParser, world::World,
    },
};

const FILENAME: &str = "samples/cow.obj";

pub fn run(width: usize, height: usize) -> Canvas {
    let light = PointLightSource::new(Point::new(-10., 10., 10.), Color::new(1., 1., 1.));

    let source = fs::read_to_string(FILENAME).unwrap();

    let mut model = ObjParser::parse_to_object(source).unwrap();
    model.transform(Matrix::rotation_y(-std::f64::consts::FRAC_PI_6).translate(-2., 0., -15.));
    let objects = vec![model];

    let world = World::new(objects, vec![light], None);

    let camera = Camera::new(width, height, consts::FRAC_PI_3);

    world.render(&camera)
}
