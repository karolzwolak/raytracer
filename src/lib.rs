pub mod approx_eq;

pub mod math;

pub mod scene;

pub use scene::{
    animation::{Animation, Animations, TransformAnimation},
    io::{
        obj_model::ObjModelParser,
        yaml::{self, YamlParser},
    },
    light::{self, point_light::PointLightSource},
    object::{
        bounding_box::{Bounded, BoundingBox},
        csg::*,
        group::*,
        material::{pattern::Pattern, Material},
        primitive::shape::*,
        *,
    },
    Scene,
};

pub mod render;
