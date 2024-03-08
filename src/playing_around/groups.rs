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
        object::{group::ObjectGroup, shape::Shape, Object},
        world::World,
    },
};

fn hexagon() -> Object {
    let corner_sphere = Object::sphere(Point::new(0., 0., -1.), 0.25);
    let cylinder = Object::with_transformation(
        Shape::Cylinder {
            y_min: 0.,
            y_max: 1.,
            closed: false,
        },
        Matrix::scaling(0.25, 1., 0.25)
            .rotate_z(-consts::FRAC_PI_2)
            .rotate_y(-consts::FRAC_PI_6)
            .translate(0., 0., -1.)
            .transformed(),
    );
    let mut hexagon_group = ObjectGroup::new(vec![cylinder.clone(), corner_sphere.clone()]);
    hexagon_group.add_bounding_box_as_obj();
    let hexagon_part = Object::with_shape(hexagon_group.into_shape());

    let mut hexagon = ObjectGroup::new(vec![hexagon_part.clone()]);

    for _ in 0..6 {
        hexagon.apply_transformation(Matrix::rotation_y(consts::FRAC_PI_3));
        hexagon.add_child(hexagon_part.clone());
    }
    hexagon.add_bounding_box_as_obj();
    Object::with_shape(hexagon.into_shape())
}

pub fn run(width: usize, height: usize) -> Canvas {
    let light = PointLightSource::new(Point::new(-10., 10., 10.), Color::new(1., 1., 1.));

    let mut hexagon = hexagon();
    hexagon.apply_transformation(
        Matrix::rotation_x(consts::FRAC_PI_3)
            .rotate_z(-consts::FRAC_PI_6)
            .translate(0., 0., -3.)
            .transformed(),
    );

    let objects = vec![hexagon];

    let world = World::new(objects, vec![light], None);
    let camera = Camera::new(width, height, consts::FRAC_PI_3);
    world.render(&camera)
}
