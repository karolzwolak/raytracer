use crate::primitive::point::Point;

use super::{color::Color, object::Object, pattern::Pattern};

#[derive(Clone, Debug)]
pub struct Material {
    pub pattern: Pattern,
    pub ambient: f64,    // [0;1]
    pub diffuse: f64,    // [0;1]
    pub specular: f64,   // [0;1]
    pub shininess: f64,  // [10;+inf) (typically up to 200.0)
    pub reflective: f64, // [0;1]
}

impl Material {
    pub fn with_pattern(pattern: Pattern) -> Self {
        Self {
            pattern,
            ..Default::default()
        }
    }

    pub fn with_color(color: Color) -> Self {
        Self::with_pattern(Pattern::Const(color))
    }

    pub fn matte_with_color(color: Color) -> Self {
        Self {
            pattern: Pattern::Const(color),
            specular: 0.05,
            shininess: 15.,
            ..Default::default()
        }
    }

    pub fn pattern(&self) -> &Pattern {
        &self.pattern
    }

    pub fn color_at(&self, point: &Point) -> Color {
        self.pattern.color_at(point)
    }

    pub fn color_at_object(&self, object: &Object, point: Point) -> Color {
        self.pattern.color_at_object(object, point)
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            pattern: Pattern::Const(Color::white()),
            ambient: 0.1,
            diffuse: 0.9,
            specular: 0.9,
            shininess: 200.,
            reflective: 0.,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_material() {
        let m = Material::default();

        assert_eq!(m.pattern(), &Pattern::Const(Color::white()));
        assert_eq!(m.ambient, 0.1);
        assert_eq!(m.diffuse, 0.9);
        assert_eq!(m.specular, 0.9);
        assert_eq!(m.shininess, 200.0);
        assert_eq!(m.reflective, 0.0);
    }
}
