use std::fs;

use crate::{render::canvas::Canvas, yaml};

const FILENAME: &str = "samples/test.yaml";

pub fn run(width: usize, height: usize) -> Canvas {
    let source = fs::read_to_string(FILENAME).unwrap();
    let (world, camera) = yaml::parse_str(&source, width, height, std::f64::consts::PI / 2.0);
    world.render(&camera)
}
