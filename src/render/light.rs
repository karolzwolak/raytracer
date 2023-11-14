use crate::primitive::point::Point;

use super::color::Color;

pub struct PointLightSource {
    position: Point,
    intensity: Color,
}

impl PointLightSource {
    pub fn new(position: Point, intensity: Color) -> Self {
        Self {
            position,
            intensity,
        }
    }

    pub fn intensity(&self) -> Color {
        self.intensity
    }

    pub fn position(&self) -> Point {
        self.position
    }
}
