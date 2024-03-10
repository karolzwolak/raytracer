use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4};

use crate::{
    primitive::{
        matrix::{Matrix, Transform},
        point::Point,
        tuple::Tuple,
        vector::Vector,
    },
    render::{
        camera::Camera,
        canvas::Canvas,
        color::Color,
        light::PointLightSource,
        material::Material,
        object::{shape::Shape, Object, PrimitiveObject},
        pattern::Pattern,
        world::World,
    },
};

pub fn scene_objects() -> Vec<Object> {
    let middle_sphere = Object::primitive(
        Shape::Sphere,
        Material {
            pattern: Pattern::Const(Color::new(0.1, 1., 0.5)),
            diffuse: 0.7,
            specular: 0.3,
            ..Default::default()
        },
        Matrix::translation(-0.5, 1., 0.5),
    );

    let mut right_sphere = Object::primitive(
        Shape::Sphere,
        Material {
            pattern: Pattern::Const(Color::new(0.5, 1., 0.1)),
            ..middle_sphere.material().clone()
        },
        Matrix::scaling_uniform(0.5)
            .translate(1.5, 0.5, -0.5)
            .transformed(),
    );

    let mut left_sphere = Object::primitive(
        Shape::Sphere,
        Material {
            pattern: Pattern::Const(Color::new(1., 0.8, 0.1)),
            diffuse: 0.7,
            specular: 0.3,
            ..Default::default()
        },
        Matrix::scaling_uniform(0.33)
            .translate(-1.5, 0.33, -0.75)
            .transformed(),
    );
    vec![middle_sphere, right_sphere, left_sphere]
}

pub fn scene_walls() -> Vec<Object> {
    let floor = Object::primitive(
        Shape::Sphere,
        Material {
            specular: 0.,
            pattern: Pattern::Const(Color::new(1., 0.9, 0.9)),
            ..Default::default()
        },
        Matrix::scaling(10., 0.01, 10.),
    );

    let left_wall = Object::primitive(
        Shape::Sphere,
        floor.material().clone(),
        Matrix::scaling(10., 0.01, 10.)
            .rotate_x(FRAC_PI_2)
            .rotate_y(-FRAC_PI_4)
            .translate(0., 0., 5.)
            .transformed(),
    );

    let right_wall = Object::primitive(
        Shape::Sphere,
        floor.material().clone(),
        Matrix::scaling(10., 0.01, 10.)
            .rotate_x(FRAC_PI_2)
            .rotate_y(FRAC_PI_4)
            .translate(0., 0., 5.)
            .transformed(),
    );

    vec![floor, left_wall, right_wall]
}

pub fn scene_lights() -> Vec<PointLightSource> {
    vec![PointLightSource::new(
        Point::new(-10., 10., -10.),
        Color::white(),
    )]
}

pub fn scene_camera(width: usize, height: usize) -> Camera {
    let from = Point::new(0., 1.5, -5.);
    let to = Point::new(0., 1., 0.);
    let up_v = Vector::new(0., 1., 0.);

    Camera::with_transformation(
        width,
        height,
        FRAC_PI_3,
        Matrix::view_tranformation(from, to, up_v),
    )
}

pub fn run(width: usize, height: usize) -> Canvas {
    let mut objects = scene_walls();
    objects.extend(scene_objects());

    let world = World::new(objects, scene_lights(), None);
    let camera = scene_camera(width, height);

    world.render(&camera)
}
