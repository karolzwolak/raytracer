use std::f64::consts;

use crate::{
    primitive::{
        matrix::{Matrix, Transform},
        point::Point,
        tuple::Tuple,
    },
    render::{canvas::Canvas, color::Color},
};

fn put_on_canvas(canvas: &mut Canvas, size: usize, x: f64, y: f64, color: Color) {
    let to_add = (size as f64) / 2.;
    canvas.write_pixel(
        (x.round() + to_add) as usize,
        (y.round() + to_add) as usize,
        color,
    );
}
pub fn run() -> Canvas {
    const SIZE: usize = 100;
    const RADIUS: f64 = (SIZE as f64) * 0.8 / 2.;

    let color = Color::blue();
    let mut canvas = Canvas::new(SIZE, SIZE);

    let first_point = Point::new(0., 0., 0.)
        .scale_uniform(0.)
        .translate(0., RADIUS, 0.)
        .transformed();
    let transformation = Matrix::identity().rotate_z(consts::FRAC_PI_6).transformed();

    let mut point = first_point;

    put_on_canvas(&mut canvas, SIZE, point.x(), point.y(), color);

    for _ in 0..11 {
        point.transform(&transformation);
        put_on_canvas(&mut canvas, SIZE, point.x(), point.y(), color);
    }

    canvas
}
