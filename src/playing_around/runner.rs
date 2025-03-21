use std::{env, fs};

use crate::playing_around::{making_patterns, planes, reflections, shadows};

use super::{
    cubes, cylinders, groups, light_and_shading, making_scene, projectiles, rays_and_spheres,
    refractions, transformations, triangles, yaml,
};

const DEFAULT_WIDTH: usize = 1200;
const DEFAULT_HEIGHT: usize = 1200;
const IMAGES_DIR: &str = "images";
const DEFAULT_CHAPTER: &str = "triangles";

pub fn run() -> Result<(), String> {
    match parse_args() {
        Some((chapter, width, height, filename)) => {
            run_with_args(chapter, width, height, &filename)
        }
        None => {
            Err("usage: cargo run -r -- <chapter>? <width>? <height>? <output file>?".to_string())
        }
    }
}

fn parse_args() -> Option<(String, usize, usize, String)> {
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

    let width = match args
        .next()
        .unwrap_or_else(|| "0".to_string())
        .parse::<usize>()
    {
        Ok(n) if n > 0 => n,
        Ok(_) => DEFAULT_WIDTH,
        _ => return None,
    };

    let height = match args
        .next()
        .unwrap_or_else(|| "0".to_string())
        .parse::<usize>()
    {
        Ok(n) if n > 0 => n,
        Ok(_) => DEFAULT_HEIGHT,
        _ => return None,
    };

    let filename = match args.next() {
        None => chapter.to_string(),
        Some(s) => s.trim().to_owned(),
    };

    Some((chapter, width, height, filename))
}

fn run_with_args(
    chapter: String,
    width: usize,
    height: usize,
    filename: &str,
) -> Result<(), String> {
    let chapter = chapter.trim();

    match fs::create_dir_all(IMAGES_DIR) {
        Ok(it) => it,
        Err(err) => {
            return Err(format!(
                "cannot create directory '{IMAGES_DIR}' because '{err}'"
            ))
        }
    };

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
        "cubes" => cubes::run(width, height),
        "cylinders" => cylinders::run(width, height),
        "groups" => groups::run(width, height),
        "triangles" => triangles::run(width, height),
        "yaml" => yaml::run(width, height),
        _ => return Err(format!("no such chapter '{chapter}'")),
    };

    let filename = format!("{IMAGES_DIR}/{filename}");
    let file = std::fs::File::create(&filename)
        .map_err(|err| format!("failed to create file '{filename}' because '{err}'"))?;
    match canvas.save_to_png(file) {
        Err(err) => Err(format!("failed to save '{filename}' because '{err}'")),
        Ok(_) => {
            println!("created file: {filename} with size {width}x{height} pixels");
            Ok(())
        }
    }
}
