use crate::{
    primitive::{point::Point, tuple::Tuple},
    render::{
        canvas::Canvas,
        color::Color,
        intersection::IntersectionCollection,
        light::{color_of_illuminated_point, PointLightSource},
        material::Material,
        object::{Object, PrimitiveObject},
        pattern::Pattern,
        ray::Ray,
    },
};
const SPHERE_RADIUS: usize = 100;
const ORIGIN_TO_SPHERE_DIST: usize = 2 * SPHERE_RADIUS;
const SPHERE_TO_CANVAS_DISTANCE: usize = 2 * SPHERE_RADIUS;
const CANVAS_SIZE: usize =
    SPHERE_RADIUS * 3 * (ORIGIN_TO_SPHERE_DIST + SPHERE_TO_CANVAS_DISTANCE) / ORIGIN_TO_SPHERE_DIST;
const CANVAS_Z: f64 = SPHERE_TO_CANVAS_DISTANCE as f64;
const ORIGIN_Z: f64 = -(ORIGIN_TO_SPHERE_DIST as f64);

pub fn run() -> Canvas {
    let color = Color::new(0.1, 0.75, 0.75);
    let bg = Color::black();
    let light_color = Color::white();

    let mut canvas = Canvas::with_color(CANVAS_SIZE, CANVAS_SIZE, bg);

    let radius = SPHERE_RADIUS as f64;
    let half_canvas: i32 = CANVAS_SIZE as i32 / 2_i32;

    let material = Material {
        pattern: Pattern::Const(color),
        ambient: 0.05,
        ..Default::default()
    };

    let mut sphere_obj = Object::from(PrimitiveObject::sphere(Point::new(0., 0., 0.), radius));
    sphere_obj.set_material(material);

    let light = PointLightSource::new(
        Point::new(-2. * radius, 2. * radius, ORIGIN_Z - radius / 2.),
        light_color,
    );

    let origin = Point::new(0., 0., ORIGIN_Z);

    for x in -half_canvas..half_canvas {
        for y in -half_canvas..half_canvas {
            let point_on_canvas = Point::new(x as f64, y as f64, CANVAS_Z);
            let direction = (point_on_canvas - origin).normalize();
            let ray = Ray::new(origin, direction);

            let intersections = IntersectionCollection::from_ray_and_obj(ray, &sphere_obj);

            let hit_point = match intersections.hit_pos() {
                None => continue,
                Some(hit_point) => hit_point,
            };

            let normal_v = sphere_obj.normal_vector_at(hit_point);
            let eye_v = -direction;

            let color =
                color_of_illuminated_point(&sphere_obj, &light, hit_point, eye_v, normal_v, 0.);

            canvas.write_pixel(
                (x + half_canvas) as usize,
                (half_canvas - y) as usize,
                color,
            );
        }
    }

    canvas
}
