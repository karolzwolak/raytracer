use crate::{
    primitive::{point::Point, tuple::Tuple, vector::Vector},
    render::{canvas::Canvas, color::Color},
};

pub struct Projectile {
    position: Point,
    velocity: Vector,
}

impl Projectile {
    pub fn new(position: Point, velocity: Vector) -> Self {
        Self { position, velocity }
    }

    pub fn update_velocity(&mut self, change: Vector) {
        self.velocity = self.velocity + change;
    }
    pub fn move_by_velocity(&mut self) {
        self.position = self.position + self.velocity;
    }
    pub fn position(&self) -> Point {
        self.position
    }
}
pub struct Environment {
    gravity: Vector,
    wind: Vector,
    projectile: Projectile,
    canvas: Canvas,
}

impl Environment {
    pub fn new(gravity: Vector, wind: Vector, projectile: Projectile, canvas: Canvas) -> Self {
        Self {
            gravity,
            wind,
            projectile,
            canvas,
        }
    }

    pub fn tick(&mut self) {
        self.projectile.move_by_velocity();
        self.projectile.update_velocity(self.wind + self.gravity);
    }

    fn projectile_pos(&self) -> Point {
        self.projectile.position()
    }

    fn projectile_pos_on_canvas(&self) -> Option<(usize, usize)> {
        if self.is_proj_out_of_bounds() {
            return None;
        }
        let pos = self.projectile_pos();

        Some((
            pos.x().round() as usize,
            self.canvas.height() - pos.y().round() as usize,
        ))
    }

    fn projectile_hit_ground(&self) -> bool {
        self.projectile_pos().y() < 0.
    }

    fn is_proj_out_of_bounds(&self) -> bool {
        let pos = self.projectile_pos();

        pos.y() < 0.5
            || pos.x() < 0.5
            || pos.y().round() as usize >= self.canvas.height()
            || pos.x().round() as usize >= self.canvas.width()
    }

    fn draw_proj_pos(&mut self, color: Color) {
        let Some(pos) = self.projectile_pos_on_canvas() else{
            return;
        };
        self.canvas.write_pixel(pos.0, pos.1, color);
    }

    pub fn run_sim(mut self, mut color: Color, fade_color: bool) -> Canvas {
        while !self.projectile_hit_ground() {
            self.draw_proj_pos(color);
            self.tick();

            if fade_color {
                color = color + Color::new(0.005, 0.005, 0.005);
            }
        }
        self.canvas
    }
}

pub fn run() -> Canvas {
    let velocity = Vector::new(0.5, 3.0, 0.).normalize() * 8.25;
    let projectile = Projectile::new(Point::new(0., 1., 0.), velocity);
    let canvas = Canvas::with_color(300, 350, Color::new(0.35, 0.35, 0.35));

    let gravity = Vector::new(0., -0.12, 0.);
    let wind = Vector::new(-0.01, 0., 0.);

    let env = Environment::new(gravity, wind, projectile, canvas);

    env.run_sim(Color::new(0.3, 0.05, 0.1), true)
}
