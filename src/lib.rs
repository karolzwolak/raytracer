pub mod approx_eq;

pub mod primitive {
    pub mod matrix4;
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
    pub mod object;
    pub mod ray;
    pub mod shape;
    pub mod world;
}

pub mod transformation;

pub mod playing_around {
    pub mod light_and_shading;
    pub mod making_scene;
    pub mod projectiles;
    pub mod rays_and_spheres;
    pub mod runner;
    pub mod transformations;
}
