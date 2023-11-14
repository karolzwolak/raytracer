use super::color::Color;

pub struct Material {
    color: Color,
    ambient: f64,   // <0;1>
    diffuse: f64,   // <0;1>
    specular: f64,  // <0;1>
    shininess: f64, // <10;inf) (typically up to 200.0)
}

impl Material {
    pub fn with_fields(
        color: Color,
        ambient: f64,
        diffuse: f64,
        specular: f64,
        shininess: f64,
    ) -> Self {
        Self {
            color,
            ambient,
            diffuse,
            specular,
            shininess,
        }
    }

    pub fn with_color(color: Color) -> Self {
        Self::with_fields(color, 0.1, 0.9, 0.9, 200.)
    }

    pub fn new() -> Self {
        Self::with_color(Color::white())
    }

    pub fn color(&self) -> Color {
        self.color
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
}

impl Default for Material {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_material() {
        let m = Material::default();

        assert_eq!(m.color(), Color::white());
        assert_eq!(m.ambient, 0.1);
        assert_eq!(m.diffuse, 0.9);
        assert_eq!(m.specular, 0.9);
        assert_eq!(m.shininess, 200.0);
    }
}
