use saphyr::Yaml;

use crate::{
    primitive::{point::Point, tuple::Tuple},
    render::{camera::Camera, color::Color, light::PointLightSource, world::World},
};

#[derive(Debug)]
pub enum YamlParseError {
    MissingField,
    InvalidField,
}

pub struct YamlParser<'a> {
    yaml: &'a Yaml,
    world: World,
    camera: Camera,
}

type YamlParseResult<T> = Result<T, YamlParseError>;
type YamlParserOutput = YamlParseResult<(World, Camera)>;

impl<'a> YamlParser<'a> {
    fn new(yaml: &'a Yaml, default_world: World, default_camera: Camera) -> Self {
        Self {
            yaml,
            world: default_world,
            camera: default_camera,
        }
    }

    fn parse_num(&self, value: &Yaml) -> YamlParseResult<f64> {
        match value {
            Yaml::Integer(value) => Ok(*value as f64),
            Yaml::Real(value) => Ok(value.parse().unwrap()),
            _ => Err(YamlParseError::InvalidField),
        }
    }

    fn parse_vec3(&self, value: &Yaml) -> YamlParseResult<(f64, f64, f64)> {
        let vec = value.as_vec().ok_or(YamlParseError::InvalidField)?;
        Ok((
            self.parse_num(&vec[0])?,
            self.parse_num(&vec[1])?,
            self.parse_num(&vec[2])?,
        ))
    }

    fn parse_color(&self, value: &Yaml) -> YamlParseResult<Color> {
        let (r, g, b) = self.parse_vec3(value)?;
        Ok(Color::new(r, g, b))
    }

    fn parse_point(&self, value: &Yaml) -> YamlParseResult<Point> {
        let (x, y, z) = self.parse_vec3(value)?;
        Ok(Point::new(x, y, z))
    }

    fn parse_ligth(&self, value: &Yaml) -> YamlParseResult<PointLightSource> {
        let at = self.parse_point(&value["at"])?;
        let intensity = self.parse_color(&value["intensity"])?;

        Ok(PointLightSource::new(at, intensity))
    }

    fn parse_add(&mut self, value: &Yaml) -> YamlParseResult<()> {
        if let Yaml::String(str_value) = value {
            match str_value.as_str() {
                "light" => {
                    let light = self.parse_ligth(value)?;
                    self.world.add_light(light);
                }
                _ => {}
            }
        };
        Ok(())
    }

    fn parse(mut self) -> YamlParserOutput {
        for yaml_obj in self.yaml.as_vec().unwrap_or(&Vec::new()) {
            if let Yaml::Hash(hash) = yaml_obj {
                let iter = hash.iter();
                for (key, value) in hash.iter() {
                    println!("{:?}: {:?}", key, value);
                    match key.as_str() {
                        Some("add") => self.parse_add(value)?,
                        _ => (),
                    }
                }
            }
        }

        Ok((self.world, self.camera))
    }
}

pub fn parse_str(source: &str, width: usize, height: usize, fov: f64) -> (World, Camera) {
    let world = World::empty();
    let camera = Camera::new(width, height, fov);

    let yaml_vec = saphyr::Yaml::load_from_str(source).unwrap();
    let Some(yaml) = yaml_vec.last() else {
        return (world, camera);
    };
    let parser = YamlParser::new(yaml, world, camera);

    parser.parse().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: usize = 600;
    const HEIGHT: usize = 800;
    const FOV: f64 = std::f64::consts::PI / 2.0;

    const COMMENT_YAML: &str = r#"#comment"#;

    const LIGHT_YAML: &str = r#"
- add: light
  at: [ 50, 100, -50 ]
  intensity: [ 1, 1, 1]
"#;

    fn parse(source: &str) -> (World, Camera) {
        parse_str(source, WIDTH, HEIGHT, FOV)
    }

    #[test]
    fn empty_yaml() {
        let (world, camera) = parse("");
        assert_eq!(world, World::empty());
        assert_eq!(camera, Camera::new(WIDTH, HEIGHT, FOV));
    }
    #[test]
    fn comments_are_supported() {
        let _ = parse("");
    }

    #[test]
    fn parse_light() {
        let (world, _) = parse(LIGHT_YAML);
        let expected_light = PointLightSource::new(Point::new(50., 100., -50.), Color::white());
        assert_eq!(world.light_sources().first(), Some(&expected_light));
    }
}
