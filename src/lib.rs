pub mod approx_eq;

pub mod core {
    pub mod color;
    pub mod matrix;
    pub mod point;
    pub mod tuple;
    pub mod vector;

    pub use color::Color;
}

pub mod scene;

pub use scene::{
    io::{
        obj_model::ObjModelParser,
        yaml::{self, YamlParser},
    },
    Scene,
};

pub mod render {
    pub mod animations;
    pub mod animator;
    pub mod camera;
    pub mod canvas;
    pub mod intersection;
    pub mod light;
    pub mod material;
    pub mod object;
    pub mod pattern;
    pub mod ray;
}
