use std::env;

use crate::playing_around::{making_patterns, planes, reflections, shadows};

use super::{
    light_and_shading, making_scene, projectiles, rays_and_spheres, refractions, transformations,
};

const SIZE: usize = 1200;
const DEFAULT_CHAPTER: &str = "reflections";

pub fn run() -> Result<(), String> {
    match parse_args() {
        Some((chapter, size, filename)) => run_with_args(chapter, size, size, &filename),
        None => Err("usage: cargo run -r -- <chapter>? <size>? <output file>?".to_string()),
    }
}

fn parse_args() -> Option<(String, usize, String)> {
    let mut args = env::args();

    // skip executable file
    args.next();

    let chapter: String = args
        .next()
        .unwrap_or_else(|| DEFAULT_CHAPTER.to_string())
        .trim()
        .to_owned();

    if chapter.contains("help") || chapter.contains("-h") {
        return None;
    }

    let size = match args
        .next()
        .unwrap_or_else(|| "0".to_string())
        .parse::<usize>()
    {
        Ok(n) if n > 0 => n,
        Ok(_) => SIZE,
        _ => return None,
    };

    let filename = match args.next() {
        None => format!("{chapter}.ppm"),
        Some(s) => s.trim().to_owned(),
    };

    Some((chapter, size, filename))
}

fn run_with_args(
    chapter: String,
    width: usize,
    height: usize,
    filename: &str,
) -> Result<(), String> {
    let chapter = chapter.trim();

    let canvas = match chapter {
        "projectiles" => projectiles::run(width, height),
        "transformations" => transformations::run(),
        "spheres" => rays_and_spheres::run(),
        "shading" => light_and_shading::run(),
        "scene" => making_scene::run(width, height),
        "shadows" => shadows::run(width, height),
        "planes" => planes::run(width, height),
        "patterns" => making_patterns::run(width, height),
        "reflections" => reflections::run(width, height),
        "refractions" => refractions::run(width, height),
        _ => return Err(format!("no such chapter '{chapter}'")),
    };

    match canvas.save_to_file(filename) {
        Err(err) => Err(format!("failed to run '{chapter}' because '{err}'")),
        Ok(_) => {
            println!("created file: {filename} with size {width}x{height} pixels");
            Ok(())
        }
    }
}
