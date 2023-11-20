use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4};

use crate::{
    primitive::{point::Point, tuple::Tuple, vector::Vector},
    render::{
        camera::Camera, color::Color, light::PointLightSource, object::Object, shape::Shape,
        world::World,
    },
    transformation::{scaling_matrix, translation_matrix, view_tranformation_matrix, Transform},
};

pub fn run() {
    let mut floor = Object::with_transformation(Shape::Sphere, scaling_matrix(10., 0.01, 10.));
    floor.material_mut().set_specular(0.);
    floor.material_mut().set_color(Color::new(1., 0.9, 0.9));

    let left_wall = Object::new(
        Shape::Sphere,
        floor.material().clone(),
        scaling_matrix(10., 0.01, 10.)
            .rotate_x(FRAC_PI_2)
            .rotate_y(-FRAC_PI_4)
            .translate(0., 0., 5.)
            .get_transformed(),
    );

    let right_wall = Object::new(
        Shape::Sphere,
        floor.material().clone(),
        scaling_matrix(10., 0.01, 10.)
            .rotate_x(FRAC_PI_2)
            .rotate_y(FRAC_PI_4)
            .translate(0., 0., 5.)
            .get_transformed(),
    );

    let mut middle_sphere =
        Object::with_transformation(Shape::Sphere, translation_matrix(-0.5, 1., 0.5));

    middle_sphere
        .material_mut()
        .set_color(Color::new(0.1, 1., 0.5));
    middle_sphere.material_mut().set_diffuse(0.7);
    middle_sphere.material_mut().set_specular(0.3);

    let mut right_sphere = Object::new(
        Shape::Sphere,
        middle_sphere.material().clone(),
        scaling_matrix(0.5, 0.5, 0.5)
            .translate(1.5, 0.5, -0.5)
            .get_transformed(),
    );
    right_sphere
        .material_mut()
        .set_color(Color::new(0.5, 1., 0.1));

    let mut left_sphere = Object::with_transformation(
        Shape::Sphere,
        scaling_matrix(0.33, 0.33, 0.33)
            .translate(-1.5, 0.33, -0.75)
            .get_transformed(),
    );

    left_sphere
        .material_mut()
        .set_color(Color::new(1., 0.8, 0.1));
    left_sphere.material_mut().set_diffuse(0.7);
    left_sphere.material_mut().set_specular(0.3);

    let light_sources = vec![PointLightSource::new(
        Point::new(-10., 10., -10.),
        Color::white(),
    )];

    let world = World::new(
        vec![
            floor,
            left_wall,
            right_wall,
            middle_sphere,
            right_sphere,
            left_sphere,
        ],
        light_sources,
    );

    let from = Point::new(0., 1.5, -5.);
    let to = Point::new(0., 1., 0.);
    let up_v = Vector::new(0., 1., 0.);

    let camera = Camera::with_transformation(
        2600,
        2600,
        FRAC_PI_3,
        view_tranformation_matrix(from, to, up_v),
    );

    world.render(&camera).save_to_file("making_scene.ppm");
}
