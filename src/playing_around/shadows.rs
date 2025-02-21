use std::f64::consts;

use crate::{
    primitive::{
        matrix::{Matrix, Transform},
        point::Point,
        tuple::Tuple,
    },
    render::{
        camera::Camera,
        canvas::Canvas,
        color::Color,
        light::PointLightSource,
        material::Material,
        object::{shape::Shape, Object, PrimitiveObject},
        world::World,
    },
};

pub fn run(width: usize, height: usize) -> Canvas {
    let wall = Object::primitive(
        Shape::Sphere,
        Material::matte_with_color(Color::new(0.4, 0.7, 0.9)),
        Matrix::scaling(50., 50., 0.1)
            .translate(0., 0., -80.)
            .transformed(),
    );

    let gray = Material::matte_with_color(Color::new(0.8, 0.8, 0.8));
    let black = Material::with_color(Color::new(0.2, 0.2, 0.2));
    let orange = Material::matte_with_color(Color::new(0.7, 0.3, 0.3));

    let x = 1.5;
    let z = -8.;

    let mut sphere1 = PrimitiveObject::sphere(Point::new(x, 0., z), 1.);
    sphere1.set_material(gray.clone());

    let mut sphere2 = PrimitiveObject::sphere(Point::new(x, 1., z), 0.7);
    sphere2.set_material(gray.clone());

    let mut sphere3 = PrimitiveObject::sphere(Point::new(x, 1.8, z), 0.4);
    sphere3.set_material(gray);

    let carrot = Object::primitive(
        Shape::Sphere,
        orange,
        Matrix::scaling(0.4, 0.1, 0.1)
            .translate(x - 0.25, 1.8, z)
            .transformed(),
    );

    let flat = Object::primitive(
        Shape::Sphere,
        black.clone(),
        Matrix::scaling(0.4, 0.05, 0.4)
            .translate(x, 2.1, z)
            .transformed(),
    );

    let cylinder = Object::primitive(
        Shape::Sphere,
        black.clone(),
        Matrix::scaling(0.25, 0.55, 0.25)
            .translate(x, 2.1, z)
            .transformed(),
    );

    let top = Object::primitive(
        Shape::Sphere,
        black,
        Matrix::scaling(0.15, 0.085, 0.15)
            .translate(x, 2.00 + 0.55, z)
            .transformed(),
    );

    let light_source = PointLightSource::new(Point::new(2. * x, 1., 4.), Color::white());

    let mut world = World::new(
        vec![
            wall,
            sphere1.into(),
            sphere2.into(),
            sphere3.into(),
            carrot,
            flat,
            cylinder,
            top,
        ],
        vec![light_source],
        None,
    );

    let camera = Camera::new(width, height, consts::FRAC_PI_4);

    world.render(&camera)
}
