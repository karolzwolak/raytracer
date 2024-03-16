use std::{f64::consts::FRAC_PI_4, f64::consts::FRAC_PI_6};

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

    let floor = Object::primitive(
        Shape::Plane,
        Material {
            pattern: Pattern::checkers(
                Color::new(0.25, 0.25, 0.25),
                Color::new(0.7, 0.7, 0.7),
                Some(
                    Matrix::rotation_y(core::f64::consts::FRAC_PI_4)
                        .scale_uniform(0.75)
                        .transformed(),
                ),
            ),
            reflectivity: 0.05,
            ..Default::default()
        },
        Matrix::identity(),
    );

    let brown = Color::new(0.4, 0.2, 0.1);
    let wood_material = Material {
        pattern: Pattern::Const(brown),
        reflectivity: 0.0,
        ..Default::default()
    };

    let leg_scaling = Matrix::identity().scale(0.08, 1., 0.08).transformed();

    let leg1 = Object::primitive(
        Shape::Cube,
        wood_material.clone(),
        leg_scaling.clone().translate(1., 0.5, 1.).transformed(),
    );

    let leg2 = Object::primitive(
        Shape::Cube,
        wood_material.clone(),
        leg_scaling.clone().translate(-1., 0.5, 1.).transformed(),
    );

    let leg3 = Object::primitive(
        Shape::Cube,
        wood_material.clone(),
        leg_scaling.clone().translate(1., 0.5, -1.).transformed(),
    );

    let leg4 = Object::primitive(
        Shape::Cube,
        wood_material.clone(),
        leg_scaling.clone().translate(-1., 0.5, -1.).transformed(),
    );

    let table_top = Object::primitive(
        Shape::Cube,
        Material {
            reflectivity: 0.05,
            ..wood_material.clone()
        },
        Matrix::identity()
            .scale(1.5, 0.05, 1.25)
            .translate(0., 1.5, 0.)
            .transformed(),
    );

    let walls_width = 5.;

    let walls = Object::primitive(
        Shape::Cube,
        Material::with_pattern(Pattern::stripe(
            Color::new(0.42, 0.55, 0.42),
            Color::new(0.32, 0.55, 0.42),
            Some(
                Matrix::scaling_uniform(0.125 / 4.)
                    .rotate_z(std::f64::consts::PI / 4.)
                    .transformed(),
            ),
        )),
        Matrix::identity()
            .scale_uniform(walls_width)
            .translate(0., 0., 0.)
            .transformed(),
    );

    let frame_scaling_ver = Matrix::identity().scale(0.05, 1.55, 0.01).transformed();
    let frame_scaling_hor = Matrix::identity().scale(0.05, 2.55, 0.01).transformed();
    let mirror_scale = Matrix::scaling(2.5, 1.5, 0.01);
    let mirror_translate = Matrix::translation(0., 2.25, -walls_width).transformed();

    let mirror_frame1 = Object::primitive(
        Shape::Cube,
        wood_material.clone(),
        frame_scaling_ver
            .clone()
            .transform_chain(&mirror_translate)
            .translate(2.5, 0., 0.)
            .transformed(),
    );

    let mirror_frame2 = Object::primitive(
        Shape::Cube,
        wood_material.clone(),
        frame_scaling_ver
            .clone()
            .transform_chain(&mirror_translate)
            .translate(-2.5, 0., 0.)
            .transformed(),
    );

    let mirror_frame3 = Object::primitive(
        Shape::Cube,
        wood_material.clone(),
        frame_scaling_hor
            .clone()
            .rotate_z(std::f64::consts::PI / 2.)
            .transform_chain(&mirror_translate)
            .translate(0., 1.5, 0.)
            .transformed(),
    );

    let mirror_frame4 = Object::primitive(
        Shape::Cube,
        wood_material,
        frame_scaling_hor
            .clone()
            .rotate_z(std::f64::consts::PI / 2.)
            .transform_chain(&mirror_translate)
            .translate(0., -1.5, 0.)
            .transformed(),
    );

    let mirror = Object::primitive(
        Shape::Cube,
        Material::mirror(),
        Matrix::identity()
            .transform_chain(&mirror_scale)
            .transform_chain(&mirror_translate)
            .translate(0., 0., 0.005)
            .transformed(),
    );

    let glass_cube = Object::primitive(
        Shape::Cube,
        Material::glass(),
        Matrix::identity()
            .scale_uniform(0.125)
            .translate(0.1, 1.65, -0.1)
            .transformed(),
    );

    let tinted_cube = Object::primitive(
        Shape::Cube,
        Material {
            pattern: Pattern::Const(Color::new(0.4, 0.2, 0.3)),
            ..Material::glass()
        },
        Matrix::identity()
            .scale(0.2, 0.05, 0.075)
            .rotate_y(FRAC_PI_4 * 0.9)
            .translate(-0.4, 1.6, 0.4)
            .transformed(),
    );

    let c1 = Color::new(0.4, 0.2, 0.3);
    let c2 = Color::new(0.2, 0.3, 0.4);

    let pattern_cube = Object::primitive(
        Shape::Cube,
        Material::with_pattern(Pattern::checkers(
            c1,
            c2,
            Some(Matrix::scaling_uniform(0.05)),
        )),
        Matrix::identity()
            .scale_uniform(0.15)
            .rotate_y(-FRAC_PI_6 * 0.75)
            .translate(0.5, 1.7, 0.6)
            .transformed(),
    );

    let mirror_cube = Object::primitive(
        Shape::Cube,
        Material::mirror(),
        Matrix::identity()
            .scale_uniform(0.15)
            .rotate_y(FRAC_PI_6 * 0.75)
            .translate(-0.75, 1.7, 0.1)
            .transformed(),
    );

    let objects = vec![
        skybox,
        floor,
        leg1,
        leg2,
        leg3,
        leg4,
        table_top,
        walls,
        mirror_frame1,
        mirror_frame2,
        mirror_frame3,
        mirror_frame4,
        mirror,
        glass_cube,
        tinted_cube,
        pattern_cube,
        mirror_cube,
    ];

    let light1 = PointLightSource::new(Point::new(-3., 5., 3.), Color::white());
    let lights = vec![light1];

    let world = World::new(objects, lights, Some(8));
    let camera = Camera::with_transformation(
        width,
        height,
        0.885,
        Matrix::view_tranformation(
            Point::new(-1., 2.6, 4.75),
            Point::new(0., 1., 0.),
            Vector::new(0., 1., 0.),
        ),
    );

    world.render(&camera)
}
