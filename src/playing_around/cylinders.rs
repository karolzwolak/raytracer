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
        object::{shape::Shape, Object},
        pattern::Pattern,
        world::World,
    },
};

pub fn run(width: usize, height: usize) -> Canvas {
    let skybox = Object::primitive(
        Shape::Cube,
        Material::with_pattern(Pattern::Const(Color::new(0.2, 0.35, 0.78))),
        Matrix::scaling_uniform(10.),
    );

    let arrow_material = Material::with_pattern(Pattern::Const(Color::new(0.8, 0.2, 0.2)));
    let arrow_rotation = Matrix::rotation_z(-std::f64::consts::PI / 4.)
        .translate(-0.5, 0., 0.)
        .transformed();

    let arrow_body = Object::primitive(
        Shape::unit_cylinder(),
        arrow_material.clone(),
        Matrix::scaling(0.1, 1., 0.1)
            .transform_chain(&arrow_rotation)
            .transformed(),
    );
    let arrow_head = Object::primitive(
        Shape::unit_cone(),
        arrow_material,
        Matrix::scaling(0.2, 0.5, 0.2)
            .translate(0., 1., 0.)
            .transform_chain(&arrow_rotation)
            .transformed(),
    );

    let ice_cream_cone = Object::primitive(
        Shape::cone(1., 0.5, false),
        Material::with_pattern(Pattern::Const(Color::new(0.67, 0.57, 0.38))),
        Matrix::scaling(0.5, 1., 0.5)
            .translate(1., -0.15, 0.)
            .transformed(),
    );

    let vanilla_scoop = Object::primitive(
        Shape::Sphere,
        Material::with_pattern(Pattern::Const(Color::new(0.95, 0.89, 0.67))),
        Matrix::scaling(0.5, 0.5, 0.5)
            .translate(1., 1., 0.)
            .transformed(),
    );

    let choc_scoop = Object::primitive(
        Shape::Sphere,
        Material::with_pattern(Pattern::Const(Color::new(0.48, 0.24, 0.))),
        Matrix::scaling(0.5, 0.5, 0.5)
            .translate(1., 1.75, 0.)
            .transformed(),
    );
    let objects = vec![
        skybox,
        arrow_body,
        arrow_head,
        ice_cream_cone,
        vanilla_scoop,
        choc_scoop,
    ];

    let light = PointLightSource::new(Point::new(-10., 10., -10.), Color::new(1., 1., 1.));

    let lights = vec![light];

    let camera = Camera::with_transformation(
        width,
        height,
        std::f64::consts::FRAC_PI_3,
        Matrix::view_tranformation(
            Point::new(0., 1.5, -5.),
            Point::new(0., 1., 0.),
            Vector::new(0., 1., 0.),
        ),
    );

    let world = World::new(objects, lights, None);

    world.render(&camera)
}
