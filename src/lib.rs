pub mod approx_eq;

pub mod primitive {
    pub mod matrix;
    pub mod point;
    pub mod tuple;
    pub mod vector;
}

pub mod render {
    pub mod camera;
    pub mod canvas;
    pub mod color;
    pub mod intersection;
    pub mod light;
    pub mod material;
    pub mod obj_parser;
    pub mod object;
    pub mod pattern;
    pub mod ray;
    pub mod world;
}

pub mod playing_around {
    pub mod cubes;
    pub mod cylinders;
    pub mod groups;
    pub mod light_and_shading;
    pub mod making_patterns;
    pub mod making_scene;
    pub mod planes;
    pub mod projectiles;
    pub mod rays_and_spheres;
    pub mod reflections;
    pub mod refractions;
    pub mod runner;
    pub mod shadows;
    pub mod transformations;
}
