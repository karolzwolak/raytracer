use crate::primitive::point::Point;

use super::{color::Color, pattern::Pattern};

#[derive(Clone, Debug)]
pub struct Material {
    pattern: Pattern,
    ambient: f64,   // <0;1>
    diffuse: f64,   // <0;1>
    specular: f64,  // <0;1>
    shininess: f64, // <10;inf) (typically up to 200.0)
}

impl Material {
    pub fn new(
        pattern: Pattern,
        ambient: f64,
        diffuse: f64,
        specular: f64,
        shininess: f64,
    ) -> Self {
        Self {
            pattern,
            ambient,
            diffuse,
            specular,
            shininess,
        }
    }

    pub fn with_pattern(pattern: Pattern) -> Self {
        Self::new(pattern, 0.1, 0.9, 0.9, 200.)
    }

    pub fn matte_with_color(color: Color) -> Self {
        Self::new(Pattern::Const(color), 0.1, 0.9, 0.05, 15.)
    }

    pub fn with_color(color: Color) -> Self {
        Self::with_pattern(Pattern::Const(color))
    }

    pub fn pattern(&self) -> &Pattern {
        &self.pattern
    }

    pub fn color_at(&self, point: &Point) -> Color {
        self.pattern.color_at(point)
    }

    pub fn set_pattern(&mut self, pattern: Pattern) {
        self.pattern = pattern;
    }

    pub fn set_color(&mut self, color: Color) {
        self.pattern = Pattern::Const(color);
    }

    pub fn ambient(&self) -> f64 {
        self.ambient
    }

    pub fn diffuse(&self) -> f64 {
        self.diffuse
    }

    pub fn specular(&self) -> f64 {
        self.specular
    }

    pub fn shininess(&self) -> f64 {
        self.shininess
    }

    pub fn set_ambient(&mut self, ambient: f64) {
        self.ambient = ambient;
    }

    pub fn set_diffuse(&mut self, diffuse: f64) {
        self.diffuse = diffuse;
    }

    pub fn set_specular(&mut self, specular: f64) {
        self.specular = specular;
    }

    pub fn set_shininess(&mut self, shininess: f64) {
        self.shininess = shininess;
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::with_color(Color::white())
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
    }
}
