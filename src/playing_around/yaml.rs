use std::fs;

use crate::{render::canvas::Canvas, yaml};

const FILENAME: &str = "samples/scenes/dragon.yml";

pub fn run(width: usize, height: usize) -> Canvas {
    let source = fs::read_to_string(FILENAME).unwrap();
    let (mut world, camera) =
        yaml::parse_str(&source, width, height, std::f64::consts::PI / 2.0).unwrap();
    world.render(&camera)
}
