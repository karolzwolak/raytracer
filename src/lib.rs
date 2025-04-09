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
    animation::{Animation, Animations, TransformAnimation},
    io::{
        obj_model::ObjModelParser,
        yaml::{self, YamlParser},
    },
    light::{self, point_light::PointLightSource},
    Scene,
};

pub mod render {
    pub mod animator;
    pub mod camera;
    pub mod canvas;
    pub mod intersection;
    pub mod material;
    pub mod object;
    pub mod pattern;
    pub mod ray;
}
