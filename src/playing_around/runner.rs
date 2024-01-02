use std::env;

use crate::playing_around::{making_patterns, planes, reflections_refractions, shadows};

use super::{light_and_shading, making_scene, projectiles, rays_and_spheres, transformations};

const SIZE: usize = 800;
const DEFAULT_CHAPTER: &str = "reflections";
const DEFAULT_FILENAME: &str = "playing_around.ppm";

pub fn run() -> Result<(), String> {
    match parse_args() {
        Some((chapter, filename)) => run_with_args(chapter, &filename),
        None => Err("usage: cargo run -r -- <chapter>? <output file>?".to_string()),
    }
}

fn parse_args() -> Option<(String, String)> {
    let mut args = env::args();

    // skip executable file
    args.next();

    let chapter: String = args.next().unwrap_or_else(|| DEFAULT_CHAPTER.to_string());

    if chapter.contains("help") || chapter.contains("-h") {
        return None;
    }

    let filename = match args.next() {
        Some(arg) => arg,
        None => DEFAULT_FILENAME.to_string(),
    };

    Some((chapter, filename))
}

fn run_with_args(chapter: String, filename: &str) -> Result<(), String> {
    let width = SIZE;
    let height = SIZE;

    let chapter = chapter.trim();

    let canvas = match chapter {
        "projectiles" => projectiles::run(width, height),
        "transformations" => transformations::run(),
        "rays and spheres" => rays_and_spheres::run(),
        "shading" => light_and_shading::run(),
        "scene" => making_scene::run(width, height),
        "shadows" => shadows::run(width, height),
        "planes" => planes::run(width, height),
        "patterns" => making_patterns::run(width, height),
        "reflections" => reflections_refractions::run(width, height),
        _ => return Err(format!("no such chapter '{chapter}'")),
    };

    match canvas.save_to_file(filename) {
        Err(err) => Err(format!("failed to run '{chapter}' because '{err}'")),
        Ok(_) => {
            println!("created file: {}", filename);
            Ok(())
        }
    }
}
