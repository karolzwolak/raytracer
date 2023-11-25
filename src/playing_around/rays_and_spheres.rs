use std::f64::consts::FRAC_PI_3;

use crate::{
    primitive::{matrix4::Matrix4, point::Point, tuple::Tuple, vector::Vector},
    render::{canvas::Canvas, color::Color, intersection::IntersecVec, object::Object, ray::Ray},
    transformation::Transform,
};

const SPHERE_RADIUS: usize = 200;
const CANVAS_SIZE: usize = SPHERE_RADIUS * 3;

pub fn run(filename: &str) -> std::io::Result<()> {
    let mut canvas = Canvas::with_color(CANVAS_SIZE, CANVAS_SIZE, Color::black());

    let radius = SPHERE_RADIUS as f64;
    let canvas_center = CANVAS_SIZE as f64 / 2.0;

    let mut sphere_obj = Object::sphere(Point::new(canvas_center, canvas_center, 0.), radius);

    sphere_obj.apply_transformation(
        Matrix4::identity_matrix()
            .scale(0.5, 0.75, 1.)
            .rotate_z(FRAC_PI_3)
            .sheare(1., 0., 0., 0., 0., 0.)
            .get_transformed(),
    );
    let ray_direction = Vector::new(0., 0., 1.);

    for x in 0..CANVAS_SIZE {
        for y in 0..CANVAS_SIZE {
            // cast ray from every point on canvas plane on (-radius - 1) z coordinate ("behind" sphere)
            let point = Point::new(x as f64, y as f64, -radius - 1.);
            let ray = Ray::new(point, ray_direction);

            if IntersecVec::from_ray_and_obj(ray, &sphere_obj).has_intersection() {
                canvas.write_pixel(x, CANVAS_SIZE - y, Color::red());
            }
        }
    }

    canvas.save_to_file(filename)
}
