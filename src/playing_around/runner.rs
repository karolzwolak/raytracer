use std::env;

use crate::playing_around::{planes, shadows};

use super::{light_and_shading, making_scene, projectiles, rays_and_spheres, transformations};

pub fn run() {
    match parse_args() {
        Ok((chapter, filename)) => run_with_args(chapter, &filename),
        Err(_) => eprintln!("error: wrong args"),
    }
}

fn parse_args() -> Result<(usize, String), String> {
    let mut args = env::args();

    args.next();
    let chapter = match args.next() {
        Some(arg) => arg.parse::<usize>().map_err(|err| err.to_string())?,
        None => 0,
    };

    let filename = match args.next() {
        Some(arg) => arg,
        None => "playing_around.ppm".to_string(),
    };

    Ok((chapter, filename))
}

fn run_with_args(chapter: usize, filename: &str) {
    let res = match chapter {
        2 => projectiles::run(filename),
        4 => transformations::run(filename),
        5 => rays_and_spheres::run(filename),
        6 => light_and_shading::run(filename),
        7 => making_scene::run(filename),
        8 => shadows::run(filename),
        9 => planes::run(filename),
        _ => planes::run(filename),
    };
    if let Err(err) = res {
        eprintln!("failed to run playing_around: {}", err);
        return;
    }
    println!("created file: {}", filename);
}
