use std::{f64::consts, io};

use crate::{
    primitive::{point::Point, tuple::Tuple},
    render::{
        camera::Camera, color::Color, light::PointLightSource, material::Material, object::Object,
        object::Shape, world::World,
    },
    transformation::{scaling_matrix, Transform},
};

pub fn run(filename: &str) -> Result<(), io::Error> {
    let wall = Object::new(
        Shape::Sphere,
        Material::matte_with_color(Color::new(0.4, 0.7, 0.9)),
        scaling_matrix(50., 50., 0.1)
            .translate(0., 0., -80.)
            .get_transformed(),
    );

    let gray = Material::matte_with_color(Color::new(0.8, 0.8, 0.8));
    let black = Material::with_color(Color::new(0.2, 0.2, 0.2));
    let orange = Material::matte_with_color(Color::new(0.7, 0.3, 0.3));

    let x = 1.5;
    let z = -8.;

    let mut sphere1 = Object::sphere(Point::new(x, 0., z), 1.);
    sphere1.set_material(gray.clone());

    let mut sphere2 = Object::sphere(Point::new(x, 1., z), 0.7);
    sphere2.set_material(gray.clone());

    let mut sphere3 = Object::sphere(Point::new(x, 1.8, z), 0.4);
    sphere3.set_material(gray);

    let carrot = Object::new(
        Shape::Sphere,
        orange,
        scaling_matrix(0.4, 0.1, 0.1)
            .translate(x - 0.25, 1.8, z)
            .get_transformed(),
    );

    let flat = Object::new(
        Shape::Sphere,
        black.clone(),
        scaling_matrix(0.4, 0.05, 0.4)
            .translate(x, 2.1, z)
            .get_transformed(),
    );

    let cylinder = Object::new(
        Shape::Sphere,
        black.clone(),
        scaling_matrix(0.25, 0.55, 0.25)
            .translate(x, 2.1, z)
            .get_transformed(),
    );

    let top = Object::new(
        Shape::Sphere,
        black,
        scaling_matrix(0.15, 0.085, 0.15)
            .translate(x, 2.00 + 0.55, z)
            .get_transformed(),
    );

    let light_source = PointLightSource::new(Point::new(2. * x, 1., 4.), Color::white());

    let world = World::new(
        vec![wall, sphere1, sphere2, sphere3, carrot, flat, cylinder, top],
        vec![light_source],
    );

    let size = 800;
    let camera = Camera::new(size, size, consts::FRAC_PI_4);

    world.render(&camera).save_to_file(filename)
}
