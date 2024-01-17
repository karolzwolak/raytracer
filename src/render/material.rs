use crate::primitive::point::Point;

use super::{color::Color, object::Object, pattern::Pattern};

pub const AIR_REFRACTIVE_INDEX: f64 = 1.0;

#[derive(Clone, Debug)]
pub struct Material {
    pub pattern: Pattern,
    pub ambient: f64,      // [0;1]
    pub diffuse: f64,      // [0;1]
    pub specular: f64,     // [0;1]
    pub shininess: f64,    // [10;+inf) (typically up to 200.0)
    pub reflectivity: f64, // [0;1]

    pub transparency: f64,     // [0;1]
    pub refractive_index: f64, // [0;1]
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
            reflectivity: 0.,
            transparency: 0.,
            refractive_index: AIR_REFRACTIVE_INDEX,
        }
    }
}

impl Material {
    pub fn glass() -> Self {
        Self {
            pattern: Pattern::Const(Color::black()),
            ambient: 0.025,
            diffuse: 0.2,
            specular: 1.,
            shininess: 300.,
            reflectivity: 0.9,
            transparency: 0.9,
            refractive_index: 1.5,
        }
    }
    pub fn air() -> Self {
        Self {
            pattern: Pattern::Const(Color::black()),
            ambient: 0.,
            diffuse: 0.,
            specular: 0.,
            shininess: 0.,
            reflectivity: 1.,
            transparency: 1.,
            refractive_index: AIR_REFRACTIVE_INDEX,
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
        assert_eq!(m.reflectivity, 0.0);
        assert_eq!(m.transparency, 0.0);
        assert_eq!(m.refractive_index, AIR_REFRACTIVE_INDEX);
    }
}
