use std::collections::HashMap;

use saphyr::Yaml;

use crate::{
    primitive::{
        matrix::{Matrix, Transform},
        point::Point,
        tuple::Tuple,
        vector::Vector,
    },
    render::{
        camera::Camera,
        color::Color,
        light::PointLightSource,
        material::Material,
        object::{shape::Shape, Object},
        pattern::Pattern,
        world::World,
    },
};

#[derive(Debug)]
pub enum YamlParseError {
    MissingField,
    InvalidField,
    UnexpectedValue,
    UnknownDefine,
}

pub struct YamlParser<'a> {
    yaml: &'a Yaml,
    world: World,
    camera: Camera,
    defines: HashMap<String, Yaml>,
}

type YamlParseResult<T> = Result<T, YamlParseError>;
type YamlParserOutput = YamlParseResult<(World, Camera)>;

macro_rules! parse_optional_field {
    ($self:ident, $yaml_body:ident, $base:ident, $field:ident) => {
        parse_optional_field!($self, $yaml_body, $base, stringify!($field), $field);
    };
    ($self:ident, $yaml_body:ident, $base:ident, $yaml_field:expr,$field_name: ident) => {
        match &$yaml_body[$yaml_field] {
            &Yaml::BadValue => {}
            val => $base.$field_name = $self.parse_num(val)?,
        }
    };
}

macro_rules! call_with_n_first_values {
    ($values:ident, 0, $func:path) => {
        $func($values)
    };
    ($values:ident, 1, $func:path) => {
        $func($values[0])
    };
    ($values:ident, 2, $func:path) => {
        $func($values[0], $values[1])
    };
    ($values:ident, 3, $func:path) => {
        $func($values[0], $values[1], $values[2])
    };
    ($values:ident, 4, $func:path) => {
        $func($values[0], $values[1], $values[2], $values[3])
    };
    ($values:ident, 6, $func:path) => {
        $func(
            $values[0], $values[1], $values[2], $values[3], $values[4], $values[5],
        )
    };
}

macro_rules! parse_transformation {
    ($code_name:ident, $values:ident, $n:tt) => {
        if $values.len() != $n {
            return Err(YamlParseError::InvalidField);
        } else {
            Ok(call_with_n_first_values!($values, $n, Matrix::$code_name))
        }
    };
}

impl<'a> YamlParser<'a> {
    fn new(yaml: &'a Yaml, default_world: World, default_camera: Camera) -> Self {
        Self {
            yaml,
            world: default_world,
            camera: default_camera,
            defines: HashMap::new(),
        }
    }

    fn parse_num(&self, value: &Yaml) -> YamlParseResult<f64> {
        match value {
            Yaml::Integer(value) => Ok(*value as f64),
            Yaml::Real(value) => Ok(value.parse().unwrap()),
            _ => Err(YamlParseError::InvalidField),
        }
    }

    fn parse_vec(&self, vector: &[Yaml]) -> YamlParseResult<Vec<f64>> {
        vector.iter().map(|v| self.parse_num(v)).collect()
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

    fn parse_vector(&self, value: &Yaml) -> YamlParseResult<Vector> {
        let (x, y, z) = self.parse_vec3(value)?;
        Ok(Vector::new(x, y, z))
    }

    fn parse_light(&self, body: &Yaml) -> YamlParseResult<PointLightSource> {
        let at = self.parse_point(&body["at"])?;
        let intensity = self.parse_color(&body["intensity"])?;

        Ok(PointLightSource::new(at, intensity))
    }

    fn parse_camera(&self, body: &Yaml) -> YamlParseResult<Camera> {
        let width = self
            .parse_num(&body["width"])
            .map(|w| w as usize)
            .unwrap_or(self.camera.target_width());
        let height = self
            .parse_num(&body["height"])
            .map(|w| w as usize)
            .unwrap_or(self.camera.target_height());
        let fov = self.parse_num(&body["field-of-view"])?;

        let from = self.parse_point(&body["from"])?;
        let to = self.parse_point(&body["to"])?;
        let up = self.parse_vector(&body["up"])?;

        let view = Matrix::view_tranformation(from, to, up);

        Ok(Camera::with_transformation(width, height, fov, view))
    }

    fn parse_material(&self, body: &Yaml) -> YamlParseResult<Material> {
        if let Some(name) = body.as_str() {
            let material = self
                .defines
                .get(name)
                .ok_or(YamlParseError::UnknownDefine)?;
            return self.parse_material(material);
        }
        let mut res = Material::default();
        match &body["color"] {
            &Yaml::BadValue => {}
            val => res.pattern = Pattern::Const(self.parse_color(val)?),
        }
        match &body["reflective"] {
            &Yaml::BadValue => {}
            val => res.reflectivity = self.parse_num(val)?,
        }

        parse_optional_field!(self, body, res, ambient);
        parse_optional_field!(self, body, res, diffuse);
        parse_optional_field!(self, body, res, specular);
        parse_optional_field!(self, body, res, shininess);
        parse_optional_field!(self, body, res, transparency);
        parse_optional_field!(self, body, res, "reflective", reflectivity);
        parse_optional_field!(self, body, res, "refractive-index", refractive_index);

        Ok(res)
    }

    fn parse_matrix(&self, body: &Yaml) -> YamlParseResult<Matrix> {
        if let Yaml::String(str_value) = body {
            return self.parse_transformation(
                self.defines
                    .get(str_value)
                    .ok_or(YamlParseError::UnknownDefine)?,
            );
        }
        let values = body.as_vec().ok_or(YamlParseError::InvalidField)?;
        if values.is_empty() {
            return Ok(Matrix::identity());
        }
        let kind = values[0].as_str().ok_or(YamlParseError::InvalidField)?;
        let vector = self.parse_vec(&values[1..])?;
        match kind {
            "translate" => parse_transformation!(translation, vector, 3),
            "scale" => parse_transformation!(scaling, vector, 3),
            "scale-uniform" => parse_transformation!(scaling_uniform, vector, 1),
            "rotate-x" => parse_transformation!(rotation_x, vector, 1),
            "rotate-y" => parse_transformation!(rotation_y, vector, 1),
            "rotate-z" => parse_transformation!(rotation_z, vector, 1),
            "shear" => parse_transformation!(shearing, vector, 6),
            _ => Err(YamlParseError::InvalidField),
        }
    }

    fn parse_transformation(&self, body: &Yaml) -> YamlParseResult<Matrix> {
        let mut res = Matrix::identity();
        for transformation in body.as_vec().unwrap_or(&Vec::new()) {
            let matrix = self.parse_matrix(transformation)?;
            res.transform(&matrix);
        }
        Ok(res)
    }

    fn parse_object(&self, body: &Yaml, obj_kind: &str) -> YamlParseResult<Object> {
        let shape = match obj_kind {
            "sphere" => Shape::Sphere,
            "cube" => Shape::Cube,
            "plane" => Shape::Plane,
            _ => unimplemented!(),
        };
        let material = self.parse_material(&body["material"])?;
        let transformation = self.parse_transformation(&body["transform"])?;
        Ok(Object::primitive(shape, material, transformation))
    }

    fn parse_add(&mut self, what: &Yaml, body: &Yaml) -> YamlParseResult<()> {
        if let Yaml::String(str_value) = what {
            match str_value.as_str() {
                "light" => {
                    let light = self.parse_light(body)?;
                    self.world.add_light(light);
                }
                "camera" => {
                    let camera = self.parse_camera(body)?;
                    self.camera = camera;
                }
                kind => {
                    let object = self.parse_object(body, kind)?;
                    self.world.add_obj(object);
                }
            }
        };
        Ok(())
    }

    fn parse_define(
        &mut self,
        name: &str,
        extends: Option<&str>,
        body: &Yaml,
    ) -> YamlParseResult<()> {
        let extends = extends.map(|s| self.defines[s].clone());
        match extends {
            Some(extend) => {
                let extend_hash = extend.as_hash().ok_or(YamlParseError::InvalidField)?;
                let body_hash = body.as_hash().ok_or(YamlParseError::InvalidField)?;
                let mut new_body = extend_hash.to_owned();
                for (key, value) in body_hash {
                    if extend_hash.contains_key(key) {
                        new_body[key] = value.clone();
                    } else {
                        new_body.insert(key.clone(), value.clone());
                    }
                }
                self.defines.insert(name.to_string(), Yaml::Hash(new_body));
            }
            None => {
                self.defines.insert(name.to_string(), body.clone());
            }
        }
        Ok(())
    }

    fn parse(mut self) -> YamlParserOutput {
        for yaml_obj in self.yaml.as_vec().unwrap_or(&Vec::new()) {
            if let Yaml::Hash(hash) = yaml_obj {
                match hash.front() {
                    Some((Yaml::String(operation), what)) => match operation.as_str() {
                        "add" => self.parse_add(what, yaml_obj)?,
                        "define" => {
                            let name = what.as_str().ok_or(YamlParseError::InvalidField)?;
                            let extends = yaml_obj["extend"].as_str();
                            let body = &yaml_obj["value"];
                            self.parse_define(name, extends, body)?;
                        }
                        _ => {}
                    },
                    _ => {
                        return Err(YamlParseError::UnexpectedValue);
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
    use crate::primitive::matrix::Transform;

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
    const CAMERA_YAML: &str = r#"
- add: camera
  width: 100
  height: 100
  field-of-view: 0.785
  from: [ -6, 6, -10 ]
  to: [ 6, 0, 6 ]
  up: [ -0.45, 1, 0 ]
"#;

    const PLANE_YAML: &str = r#"
- add: plane
  material:
    color: [ 1, 1, 1 ]
    ambient: 1
    diffuse: 0
    specular: 0
  transform:
    - [ rotate-x, 1.5707963267948966 ] # pi/2
    - [ translate, 0, 0, 500 ]
"#;
    const SPHERE_YAML: &str = r#"
- add: sphere
  material:
    color: [ 0.373, 0.404, 0.550 ]
    diffuse: 0.2
    ambient: 0.0
    specular: 1.0
    shininess: 200
    reflective: 0.7
    transparency: 0.7
    refractive-index: 1.5
"#;

    const DEFINE_TRANSFORMS_YAML: &str = r#"
- define: standard-transform
  value:
    - [ translate, 1, -1, 1 ]
    - [ scale, 0.5, 0.5, 0.5 ]
- define: large-object
  value:
    - standard-transform
    - [ scale-uniform, 4 ]
- add: sphere
  transform: 
    - standard-transform
- add: cube
  transform: 
    - large-object
"#;
    const DEFINE_MATERIALS_YAML: &str = r#"
- define: white-material
  value:
    color: [ 1, 1, 1 ]
    diffuse: 0.7
- define: blue-material
  extend: white-material
  value:
    color: [ 0, 0, 1 ]
- add: sphere
  material: white-material
- add: cube
  material: blue-material
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

    #[test]
    fn parse_camera() {
        let (_, camera) = parse(CAMERA_YAML);
        let view = Matrix::view_tranformation(
            Point::new(-6., 6., -10.),
            Point::new(6., 0., 6.),
            Vector::new(-0.45, 1., 0.),
        );
        let expected_camera = Camera::with_transformation(100, 100, 0.785, view);
        assert_eq!(camera, expected_camera);
    }

    #[test]
    fn parse_plane() {
        let (world, _) = parse(PLANE_YAML);
        let expected_material = Material {
            pattern: Pattern::Const(Color::white()),
            ambient: 1.,
            diffuse: 0.,
            specular: 0.,
            ..Material::default()
        };
        let expected_transformation = Matrix::rotation_x(std::f64::consts::PI / 2.0)
            .translate(0., 0., 500.)
            .transformed();
        let expected_object =
            Object::primitive(Shape::Plane, expected_material, expected_transformation);
        assert_eq!(world.objects(), vec![expected_object]);
    }

    #[test]
    fn parse_sphere() {
        let (world, _) = parse(SPHERE_YAML);
        let expected_material = Material {
            pattern: Pattern::Const(Color::new(0.373, 0.404, 0.550)),
            ambient: 0.,
            diffuse: 0.2,
            specular: 1.,
            shininess: 200.,
            reflectivity: 0.7,
            transparency: 0.7,
            refractive_index: 1.5,
        };
        let expected_object =
            Object::primitive(Shape::Sphere, expected_material, Matrix::identity());

        assert_eq!(world.objects(), vec![expected_object]);
    }

    #[test]
    fn parse_define_materials() {
        let (world, _) = parse(DEFINE_MATERIALS_YAML);
        let white_material = Material {
            pattern: Pattern::Const(Color::white()),
            diffuse: 0.7,
            ..Material::default()
        };
        let blue_material = Material {
            pattern: Pattern::Const(Color::blue()),
            diffuse: 0.7,
            ..white_material
        };
        let white_sphere = Object::primitive(Shape::Sphere, white_material, Matrix::identity());
        let blue_cube = Object::primitive(Shape::Cube, blue_material, Matrix::identity());
        let expected_objects = vec![white_sphere, blue_cube];

        assert_eq!(world.objects(), expected_objects);
    }

    #[test]
    fn parse_define_transforms() {
        let (world, _) = parse(DEFINE_TRANSFORMS_YAML);
        let standard_transform = Matrix::translation(1., -1., 1.)
            .scale(0.5, 0.5, 0.5)
            .transformed();
        let large_object_transform = standard_transform.clone().scale_uniform(4.).transformed();
        let sphere = Object::primitive(Shape::Sphere, Material::default(), standard_transform);
        let cube = Object::primitive(Shape::Cube, Material::default(), large_object_transform);
        let expected_objects = vec![sphere, cube];
        assert_eq!(world.objects(), expected_objects);
    }
}
