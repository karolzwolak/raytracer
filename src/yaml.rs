use std::{collections::HashMap, fmt::Display};

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
        obj_parser::ObjParser,
        object::{
            cone::Cone, cylinder::Cylinder, group::ObjectGroup, shape::Shape,
            smooth_triangle::SmoothTriangle, triangle::Triangle, Object,
        },
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
    FileReadError,
    ObjParsingError,
    InternalError,
    EmptyGroup, // empty groups don't make sense, it has to a mistake, so we return an error
}

impl Display for YamlParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

const PREDEFINED_DEFINES: &str = r#"
- define: PI
  value: 3.141592653589793
- define: FRAC_PI_2
  value: 1.5707963267948966
- define: FRAC_PI_3
  value: 1.0471975511965979
- define: FRAC_PI_4
  value: 0.7853981633974483
- define: FRAC_PI_6
  value: 0.5235987755982989
- define: FRAC_1_SQRT_2
  value: 0.7071067811865476

- define: GLASS_MATERIAL
  value:
    color: [ 0, 0, 0 ]
    ambient: 0.025
    diffuse: 0.2
    specular: 1.0
    shininess: 300.0
    reflective: 0.9
    transparency: 0.9
    refractive-index: 1.5

- define: MIRROR_MATERIAL
  extend: GLASS_MATERIAL
  value:
    reflective: 0.98
    transparency: 0

- define: AIR_MATERIAL
  value:
    color: [ 0, 0, 0 ]
    ambient: 0
    diffuse: 0
    specular: 0
    shininess: 0
    reflectivity: 1
    transparency: 1
    refractive-index: 1.0

- define: SCENE_LIGHT
  value:
    add: light
    at: [ -10, 10, -10 ]
    intensity: [ 1, 1, 1 ]

- define: SCENE_CAMERA
  value:
    add: camera
    from: [ 0, 1.5, -5 ]
    to: [ 0, 1, 0 ]
    up: [ 0, 1, 0 ]
    fov: 1.0471975511965979 # pi / 3
"#;

// TODO: Actual errors
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
    fn with_world_and_camera(yaml: &'a Yaml, default_world: World, default_camera: Camera) -> Self {
        let parsed = saphyr::Yaml::load_from_str(PREDEFINED_DEFINES).unwrap();
        let predefined = parsed.first().unwrap();

        let mut predefined_parser = YamlParser {
            yaml: predefined,
            world: World::empty(),
            camera: Camera::new(1, 1, 1.),
            defines: HashMap::new(),
        };
        predefined_parser
            .parse()
            .expect("Error parsing predefined defines");
        Self {
            yaml,
            world: default_world,
            camera: default_camera,
            defines: predefined_parser.defines,
        }
    }

    fn new(yaml: &'a Yaml) -> Self {
        Self::with_world_and_camera(yaml, World::empty(), Camera::new(1, 1, 1.))
    }

    fn parse_num(&self, value: &Yaml) -> YamlParseResult<f64> {
        match value {
            Yaml::Integer(value) => Ok(*value as f64),
            Yaml::Real(value) => Ok(value.parse().unwrap()),
            Yaml::String(name) => {
                let value = self
                    .defines
                    .get(name)
                    .ok_or(YamlParseError::UnknownDefine)?;
                self.parse_num(value)
            }
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
        if let Yaml::String(str_value) = value {
            let color = self
                .defines
                .get(str_value)
                .ok_or(YamlParseError::UnknownDefine)?;
            return self.parse_color(color);
        }
        let (r, g, b) = self.parse_vec3(value)?;
        Ok(Color::new(r, g, b))
    }

    fn parse_pattern(&self, body: &Yaml) -> YamlParseResult<Pattern> {
        let kind = body["type"].as_str().ok_or(YamlParseError::InvalidField)?;
        let colors = body["colors"]
            .as_vec()
            .ok_or(YamlParseError::InvalidField)?
            .iter()
            .map(|c| self.parse_color(c))
            .collect::<YamlParseResult<Vec<Color>>>()?;

        if colors.len() != 2 {
            return Err(YamlParseError::InvalidField);
        }
        let transform = match &body["transform"] {
            &Yaml::BadValue => None,
            val => Some(self.parse_transformation(val)?),
        };

        Ok(match kind {
            "stripe" | "stripes" => Pattern::stripe(colors[0], colors[1], transform),
            "gradient" => Pattern::gradient(colors[0], colors[1], transform),
            "checkers" => Pattern::checkers(colors[0], colors[1], transform),
            "ring" => Pattern::ring(colors[0], colors[1], transform),
            "color" => Pattern::Const(colors[0]),
            _ => return Err(YamlParseError::InvalidField),
        })
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

    fn parse_bool(&self, value: &Yaml) -> YamlParseResult<bool> {
        value.as_bool().ok_or(YamlParseError::InvalidField)
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
        let fov_body = match &body["field-of-view"] {
            &Yaml::BadValue => &body["fov"],
            val => val,
        };
        let fov = self.parse_num(fov_body)?;

        let from = self.parse_point(&body["from"])?;
        let to = self.parse_point(&body["to"])?;
        let up = self.parse_vector(&body["up"])?;

        let view = Matrix::view_tranformation(from, to, up);

        Ok(Camera::with_transformation(width, height, fov, view))
    }

    fn parse_material(&self, body: &Yaml) -> YamlParseResult<Material> {
        match *body {
            Yaml::BadValue => {
                return Ok(Material::default());
            }
            Yaml::String(ref name) => {
                let material = self
                    .defines
                    .get(name)
                    .ok_or(YamlParseError::UnknownDefine)?;
                return self.parse_material(material);
            }
            _ => {}
        }
        let mut res = Material::default();

        match (&body["color"], &body["pattern"]) {
            (&Yaml::BadValue, &Yaml::BadValue) => {}
            (val, &Yaml::BadValue) => res.pattern = Pattern::Const(self.parse_color(val)?),
            (&Yaml::BadValue, val) => res.pattern = self.parse_pattern(val)?,
            (_, _) => return Err(YamlParseError::InvalidField),
        }

        parse_optional_field!(self, body, res, ambient);
        parse_optional_field!(self, body, res, diffuse);
        parse_optional_field!(self, body, res, specular);
        parse_optional_field!(self, body, res, shininess);
        parse_optional_field!(self, body, res, transparency);

        parse_optional_field!(self, body, res, "reflective", reflectivity);
        parse_optional_field!(self, body, res, reflectivity); // not a mistake, we permit both
                                                              // names
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

    fn parse_group(&self, body: &Yaml) -> YamlParseResult<ObjectGroup> {
        let children_yaml = body["children"]
            .as_vec()
            .ok_or(YamlParseError::EmptyGroup)?;
        if children_yaml.is_empty() {
            return Err(YamlParseError::EmptyGroup);
        }
        let mut children = Vec::new();
        for child in children_yaml {
            match child.as_hash() {
                Some(hash) => {
                    let pair = hash.front().ok_or(YamlParseError::InvalidField)?;
                    match pair {
                        (Yaml::String(op), Yaml::String(kind)) if op == "add" => {
                            let object = self.parse_object(child, kind)?;
                            children.push(object);
                        }
                        _ => return Err(YamlParseError::UnexpectedValue),
                    }
                }
                None => return Err(YamlParseError::UnexpectedValue),
            }
        }
        Ok(ObjectGroup::new(children))
    }

    fn parse_obj_model(&self, body: &Yaml) -> YamlParseResult<ObjectGroup> {
        let file = body["file"].as_str().ok_or(YamlParseError::MissingField)?;
        let data = std::fs::read_to_string(file).map_err(|_| YamlParseError::FileReadError)?;
        let parser = ObjParser::new();
        parser
            .parse(data)
            .map_err(|_| YamlParseError::ObjParsingError)
    }

    fn parse_object(&self, body: &Yaml, obj_kind: &str) -> YamlParseResult<Object> {
        let shape = match obj_kind {
            "group" | "obj" => {
                let mut res = match obj_kind {
                    "group" => self.parse_group(body),
                    "obj" => self.parse_obj_model(body),
                    _ => unreachable!(),
                }?;
                match &body["material"] {
                    &Yaml::BadValue => {}
                    val => {
                        let material = self.parse_material(val)?;
                        res.set_material(material);
                    }
                }
                match &body["transform"] {
                    &Yaml::BadValue => {}
                    val => res.transform(&self.parse_transformation(val)?),
                }
                return Ok(Object::Group(res));
            }
            "sphere" => Shape::Sphere,
            "cube" => Shape::Cube,
            "plane" => Shape::Plane,
            "cylinder" => {
                let min = self.parse_num(&body["min"])?;
                let max = self.parse_num(&body["max"])?;
                let closed = self.parse_bool(&body["closed"])?;
                Shape::Cylinder(Cylinder::new(min, max, closed))
            }
            "cone" => {
                let min = self.parse_num(&body["min"])?;
                let max = self.parse_num(&body["max"])?;
                let closed = self.parse_bool(&body["closed"])?;
                Shape::Cone(Cone::new(min, max, closed))
            }
            "triangle" => {
                let p1 = self.parse_point(&body["p1"])?;
                let p2 = self.parse_point(&body["p2"])?;
                let p3 = self.parse_point(&body["p3"])?;
                Shape::Triangle(Triangle::new(p1, p2, p3))
            }
            "smooth-triangle" => {
                let p1 = self.parse_point(&body["p1"])?;
                let p2 = self.parse_point(&body["p2"])?;
                let p3 = self.parse_point(&body["p3"])?;
                let n1 = self.parse_vector(&body["n1"])?;
                let n2 = self.parse_vector(&body["n2"])?;
                let n3 = self.parse_vector(&body["n3"])?;
                Shape::SmoothTriangle(SmoothTriangle::new(p1, p2, p3, n1, n2, n3))
            }
            name => {
                if let Some(def) = self.defines.get(name) {
                    let body = self.merge_use_define(name, def, body)?;
                    let what = &body["add"].as_str().ok_or(YamlParseError::InvalidField)?;
                    return self.parse_object(&body, what);
                }
                return Err(YamlParseError::InvalidField);
            }
        };
        let material = self.parse_material(&body["material"])?;
        let transformation = self.parse_transformation(&body["transform"])?;
        Ok(Object::primitive(shape, material, transformation))
    }

    fn parse_world(&mut self, body: &Yaml) -> YamlParseResult<()> {
        match &body["supersampling-level"] {
            &Yaml::BadValue => {}
            val => self
                .world
                .set_supersampling_level(self.parse_num(val)? as usize),
        }
        match &body["max-reflective-depth"] {
            &Yaml::BadValue => {}
            val => self
                .world
                .set_max_recursive_depth(self.parse_num(val)? as usize),
        }
        match &body["use-shadow-intensity"] {
            &Yaml::BadValue => {}
            val => self.world.set_use_shadow_intensity(self.parse_bool(val)?),
        }
        Ok(())
    }

    fn parse_add(&mut self, what: &str, body: &Yaml) -> YamlParseResult<()> {
        match what {
            "light" => {
                let light = self.parse_light(body)?;
                self.world.add_light(light);
            }
            "camera" => {
                let camera = self.parse_camera(body)?;
                self.camera = camera;
            }
            "world" => {
                self.parse_world(body)?;
            }
            "group" | "obj" | "sphere" | "cube" | "plane" | "cylinder" | "cone" | "triangle"
            | "smooth-triangle" => {
                let object = self.parse_object(body, what)?;
                self.world.add_obj(object);
            }
            name => {
                if let Some(def) = self.defines.get(name) {
                    let body = self.merge_use_define(name, def, body)?;
                    return self.parse_operation(&body);
                }
            }
        }
        Ok(())
    }

    fn merge_hash(
        &self,
        name: &str,
        define_hash: &saphyr::Hash,
        use_hash: &saphyr::Hash,
    ) -> YamlParseResult<Yaml> {
        let mut new_hash = define_hash.clone();

        for (key, value) in use_hash {
            if define_hash.contains_key(key) {
                let key_str = key.as_str().ok_or(YamlParseError::InvalidField)?;
                new_hash[key] = self.merge_yaml(name, key_str, &define_hash[key], value)?;
            } else {
                new_hash.insert(key.clone(), value.clone());
            }
        }
        Ok(Yaml::Hash(new_hash))
    }

    fn merge_yaml(
        &self,
        name: &str,
        key_str: &str,
        define_yaml: &Yaml,
        use_yaml: &Yaml,
    ) -> YamlParseResult<Yaml> {
        Ok(match (define_yaml, use_yaml) {
            (&Yaml::BadValue, val) => val.clone(),
            (val, &Yaml::BadValue) => val.clone(),
            (Yaml::Hash(ref define_hash), Yaml::Hash(ref use_hash)) => {
                return self.merge_hash(name, define_hash, use_hash);
            }
            (Yaml::Array(ref define_array), Yaml::Array(ref use_array))
                if key_str == "transform" =>
            {
                let mut new_array = define_array.clone();
                new_array.extend(use_array.iter().cloned());
                Yaml::Array(new_array)
            }
            (Yaml::String(define_str), Yaml::String(use_str)) => Yaml::String(if use_str == name {
                define_str.clone()
            } else {
                use_str.clone()
            }),
            (_, val) => val.clone(),
        })
    }

    fn merge_use_define(
        &self,
        define_name: &str,
        define_body: &Yaml,
        body: &Yaml,
    ) -> YamlParseResult<Yaml> {
        let extend_hash = define_body.as_hash().ok_or(YamlParseError::InvalidField)?;
        let body_hash = body.as_hash().ok_or(YamlParseError::InvalidField)?;
        self.merge_hash(define_name, extend_hash, body_hash)
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
                self.defines.insert(
                    name.to_string(),
                    self.merge_use_define(name, &extend, body)?,
                );
            }
            None => {
                self.defines.insert(name.to_string(), body.clone());
            }
        }
        Ok(())
    }

    fn parse_operation(&mut self, yaml_obj: &Yaml) -> YamlParseResult<()> {
        if let Yaml::Hash(hash) = yaml_obj {
            match hash.front() {
                Some((Yaml::String(operation), Yaml::String(what))) => match operation.as_str() {
                    "add" => self.parse_add(what, yaml_obj)?,
                    "define" => {
                        let extends = yaml_obj["extend"].as_str();
                        let body = &yaml_obj["value"];
                        self.parse_define(what, extends, body)?;
                    }
                    _ => {}
                },
                _ => {
                    return Err(YamlParseError::UnexpectedValue);
                }
            }
        }
        Ok(())
    }

    fn parse(&mut self) -> YamlParseResult<()> {
        for yaml_obj in self.yaml.as_vec().unwrap_or(&Vec::new()) {
            self.parse_operation(yaml_obj)?;
        }
        Ok(())
    }

    fn parse_consume(mut self) -> YamlParserOutput {
        self.parse()?;
        Ok((self.world, self.camera))
    }
}

pub fn parse_str(source: &str, width: usize, height: usize, fov: f64) -> YamlParserOutput {
    let world = World::empty();
    let camera = Camera::new(width, height, fov);

    let yaml_vec = saphyr::Yaml::load_from_str(source).unwrap();
    let Some(yaml) = yaml_vec.last() else {
        return Ok((world, camera));
    };
    let parser = YamlParser::with_world_and_camera(yaml, world, camera);

    parser.parse_consume()
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_3;

    use crate::{
        primitive::matrix::Transform,
        render::object::{cylinder::Cylinder, smooth_triangle::SmoothTriangle, triangle::Triangle},
    };

    use super::*;

    const WIDTH: usize = 600;
    const HEIGHT: usize = 800;
    const FOV: f64 = std::f64::consts::PI / 2.0;

    fn parse(source: &str) -> (World, Camera) {
        parse_str(source, WIDTH, HEIGHT, FOV).unwrap()
    }

    #[test]
    fn empty_yaml() {
        let (world, camera) = parse("");
        assert_eq!(world, World::empty());
        assert_eq!(camera, Camera::new(WIDTH, HEIGHT, FOV));
    }
    #[test]
    fn comments_are_supported() {
        let source = "#comment";
        let _ = parse(source);
    }

    #[test]
    fn parse_light() {
        const LIGHT_YAML: &str = r#"
- add: light
  at: [ 50, 100, -50 ]
  intensity: [ 1, 1, 1]
"#;
        let (world, _) = parse(LIGHT_YAML);
        let expected_light = PointLightSource::new(Point::new(50., 100., -50.), Color::white());
        assert_eq!(world.light_sources().first(), Some(&expected_light));
    }

    #[test]
    fn parse_camera() {
        const CAMERA_YAML: &str = r#"
- add: camera
  width: 100
  height: 100
  field-of-view: 0.785
  from: [ -6, 6, -10 ]
  to: [ 6, 0, 6 ]
  up: [ -0.45, 1, 0 ]
"#;
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
    fn parse_world() {
        const WORLD_YAML: &str = r#"
- add: world
  max-reflective-depth: 4
  supersampling-level: 3
  use-shadow-intensity: false
"#;
        let (world, _) = parse(WORLD_YAML);
        assert_eq!(world.max_recursive_depth(), 4);
        assert_eq!(world.supersampling_level(), 3);
        assert!(!world.use_shadow_intensity());
    }

    #[test]
    fn parse_plane() {
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
    fn objects_have_default_material_and_transformation() {
        const DEFAULT_SPHERE_YAML: &str = r#"
- add: sphere
"#;
        let (world, _) = parse(DEFAULT_SPHERE_YAML);
        let sphere = Object::primitive(Shape::Sphere, Material::default(), Matrix::identity());
        assert_eq!(world.objects(), vec![sphere]);
    }

    #[test]
    fn parse_sphere() {
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

    #[test]
    fn parse_group() {
        const GROUP_YAML: &str = r#"
- add: group
  transform:
    - [translate, 1, 1, 1]
  children:
    - add: sphere
      material:
        color: [ 1, 0, 0 ]
    - add: cube
      material:
        color: [ 0, 1, 0 ]
"#;
        let (world, _) = parse(GROUP_YAML);
        let red_material = Material {
            pattern: Pattern::Const(Color::red()),
            ..Material::default()
        };
        let green_material = Material {
            pattern: Pattern::Const(Color::green()),
            ..Material::default()
        };
        let red_sphere = Object::primitive(Shape::Sphere, red_material, Matrix::identity());
        let green_cube = Object::primitive(Shape::Cube, green_material, Matrix::identity());
        let transformation = Matrix::translation(1., 1., 1.);

        let group = ObjectGroup::with_transformations(vec![red_sphere, green_cube], transformation);

        assert_eq!(world.objects(), vec![Object::Group(group)]);
    }

    #[test]
    fn empty_groups_cause_error() {
        let sources = vec![
            r#"
- add: group
  children:
"#,
            r#"
- add: group
"#,
            r#"
- add: group
  children: []
"#,
        ];
        for source in sources {
            let res = parse_str(source, WIDTH, HEIGHT, FOV);
            assert!(matches!(res, Err(YamlParseError::EmptyGroup)));
        }
    }

    #[test]
    fn parse_obj() {
        const OBJ_YAML: &str = r#"
- add: obj
  file: samples/obj/teapot-low.obj
"#;
        let (world, _) = parse(OBJ_YAML);
        let parser = ObjParser::new();
        let path = "samples/obj/teapot-low.obj";
        let data = std::fs::read_to_string(path).unwrap();
        let expected_group = parser.parse(data).unwrap();
        assert_eq!(world.objects(), vec![Object::Group(expected_group)]);
    }

    #[test]
    fn parse_use_define_in_add() {
        const USE_DEFINE_IN_ADD_YAML: &str = r#"
- define: defined_cube
  value:
    add: cube
- add: defined_cube
  material:
    color: [ 1, 0, 0 ]
"#;
        let (world, _) = parse(USE_DEFINE_IN_ADD_YAML);
        let red_material = Material {
            pattern: Pattern::Const(Color::red()),
            ..Material::default()
        };
        let red_cube = Object::primitive(Shape::Cube, red_material, Matrix::identity());
        let expected_objects = vec![red_cube];
        assert_eq!(world.objects(), expected_objects);
    }

    #[test]
    fn parse_define_color() {
        const DEFINE_COLOR_YAML: &str = r#"
- define: red
  value: [ 1, 0, 0 ]
- add: sphere
  material:
    color: red
"#;

        let (world, _) = parse(DEFINE_COLOR_YAML);
        let red_material = Material {
            pattern: Pattern::Const(Color::red()),
            ..Material::default()
        };
        let red_sphere = Object::primitive(Shape::Sphere, red_material, Matrix::identity());
        let expected_objects = vec![red_sphere];
        assert_eq!(world.objects(), expected_objects);
    }

    #[test]
    fn parse_cylinder() {
        const CYLINDER_YAML: &str = r#"
- add: cylinder
  min: 1
  max: 5
  closed: true
"#;

        let (world, _) = parse(CYLINDER_YAML);
        let cylinder_shape = Cylinder::new(1., 5., true);
        let expected_object = Object::primitive_with_shape(Shape::Cylinder(cylinder_shape));
        assert_eq!(world.objects(), vec![expected_object]);
    }

    #[test]
    fn parse_cone() {
        const CONE_YAML: &str = r#"
- add: cone
  min: 1
  max: 5
  closed: true
"#;

        let (world, _) = parse(CONE_YAML);
        let cylinder_shape = Cone::new(1., 5., true);
        let expected_object = Object::primitive_with_shape(Shape::Cone(cylinder_shape));
        assert_eq!(world.objects(), vec![expected_object]);
    }

    #[test]
    fn parse_triangle() {
        const TRIANGLE_YAML: &str = r#"
- add: triangle
  p1: [ 0, 1, 0 ]
  p2: [ -1, 0, 0 ]
  p3: [ 1, 0, 0 ]
"#;

        let (world, _) = parse(TRIANGLE_YAML);
        let triangle = Object::primitive_with_shape(Shape::Triangle(Triangle::new(
            Point::new(0., 1., 0.),
            Point::new(-1., 0., 0.),
            Point::new(1., 0., 0.),
        )));
        let expected_objects = vec![triangle];
        assert_eq!(world.objects(), expected_objects);
    }
    #[test]
    fn parse_smooth_triangle() {
        const SMOOTH_TRIANGLE_YAML: &str = r#"
- add: smooth-triangle
  p1: [ 0, 1, 0 ]
  p2: [ -1, 0, 0 ]
  p3: [ 1, 0, 0 ]
  n1: [ 0, 1, 0 ]
  n2: [ -1, 0, 0 ]
  n3: [ 1, 0, 0 ]
"#;

        let (world, _) = parse(SMOOTH_TRIANGLE_YAML);
        let triangle = Object::primitive_with_shape(Shape::SmoothTriangle(SmoothTriangle::new(
            Point::new(0., 1., 0.),
            Point::new(-1., 0., 0.),
            Point::new(1., 0., 0.),
            Vector::new(0., 1., 0.),
            Vector::new(-1., 0., 0.),
            Vector::new(1., 0., 0.),
        )));
        let expected_objects = vec![triangle];
        assert_eq!(world.objects(), expected_objects);
    }

    #[test]
    fn parse_patterns() {
        const PATTERNS_YAML: &str = r#"
- define: red
  value: [ 1, 0, 0 ]
- define: green
  value: [ 0, 1, 0 ]

- define: transformation
  value:
    - [ scale, 0.1, 0.1, 0.1 ]

- add: cube
  material:
    pattern:
      type: stripe
      colors:
        - red
        - green
      transform:
        - transformation

- add: cube
  material:
    pattern:
      type: gradient
      colors:
        - red
        - green
      transform:
        - transformation

- add: cube
  material:
    pattern:
      type: ring
      colors:
        - red
        - green
      transform:
        - transformation

- add: cube
  material:
    pattern:
      type: checkers
      colors:
        - red
        - green
      transform:
        - transformation
"#;
        let (world, _) = parse(PATTERNS_YAML);
        let red = Color::red();
        let green = Color::green();
        let transformation = Matrix::scaling_uniform(0.1);
        let stripe = Material::with_pattern(Pattern::stripe(red, green, Some(transformation)));
        let gradient = Material::with_pattern(Pattern::gradient(red, green, Some(transformation)));
        let ring = Material::with_pattern(Pattern::ring(red, green, Some(transformation)));
        let checkers = Material::with_pattern(Pattern::checkers(red, green, Some(transformation)));
        let expected_objects = vec![
            Object::primitive(Shape::Cube, stripe, Matrix::identity()),
            Object::primitive(Shape::Cube, gradient, Matrix::identity()),
            Object::primitive(Shape::Cube, ring, Matrix::identity()),
            Object::primitive(Shape::Cube, checkers, Matrix::identity()),
        ];
        assert_eq!(world.objects(), expected_objects);
    }

    #[test]
    fn predefined_materials() {
        let source = r#"
- add: sphere
  material: GLASS_MATERIAL
- add: sphere
  material: MIRROR_MATERIAL
- add: sphere
  material: AIR_MATERIAL
"#;
        let (world, _) = parse(source);
        let glass_sphere = Object::primitive(Shape::Sphere, Material::glass(), Matrix::identity());
        let mirror_sphere =
            Object::primitive(Shape::Sphere, Material::mirror(), Matrix::identity());
        let air_sphere = Object::primitive(Shape::Sphere, Material::air(), Matrix::identity());
        let expected_objects = vec![glass_sphere, mirror_sphere, air_sphere];
        assert_eq!(world.objects(), expected_objects);
    }

    #[test]
    fn predefined_scene_light_and_camera() {
        let source = r#"
- add: SCENE_LIGHT
- add: SCENE_CAMERA
"#;
        let (world, camera) = parse(source);
        let expected_camera = Camera::with_transformation(
            WIDTH,
            HEIGHT,
            FRAC_PI_3,
            Matrix::view_tranformation(
                Point::new(0., 1.5, -5.),
                Point::new(0., 1., 0.),
                Vector::new(0., 1., 0.),
            ),
        );
        let light = PointLightSource::new(Point::new(-10., 10., -10.), Color::white());
        assert_eq!(camera, expected_camera);
        assert_eq!(world.light_sources().first(), Some(&light));
    }

    #[test]
    fn predefined_pi_constants() {
        let source = r#"
- add: sphere
  material:
    color: [ 1, 1, 1 ]
    ambient: PI
    diffuse: FRAC_PI_2
    specular: FRAC_PI_3
    shininess: FRAC_PI_4
    reflective: FRAC_PI_6
    transparency: FRAC_1_SQRT_2
"#;
        let (world, _) = parse(source);
        let material = Material {
            ambient: std::f64::consts::PI,
            diffuse: std::f64::consts::FRAC_PI_2,
            specular: std::f64::consts::FRAC_PI_3,
            shininess: std::f64::consts::FRAC_PI_4,
            reflectivity: std::f64::consts::FRAC_PI_6,
            transparency: std::f64::consts::FRAC_1_SQRT_2,
            ..Material::default()
        };
        let sphere = Object::primitive(Shape::Sphere, material, Matrix::identity());
        assert_eq!(world.objects(), vec![sphere]);
    }

    #[test]
    fn merging_material_define() {
        let source = r#"
- define: red-material
  value:
    color: [ 1, 0, 0 ]
    ambient: 0.5
    diffuse: 1
- define: green-material
  extend: red-material
  value:
    color: [ 0, 1, 0 ]
    ambient: 1
- add: sphere
  material: green-material
"#;
        let (world, _) = parse(source);
        let green_material = Material {
            pattern: Pattern::Const(Color::green()),
            ambient: 1.,
            diffuse: 1.,
            ..Material::default()
        };
        let sphere = Object::primitive(Shape::Sphere, green_material, Matrix::identity());
        assert_eq!(world.objects(), vec![sphere]);
    }

    #[test]
    fn merging_transform_define() {
        let source = r#"
- define: rotation-x
  value:
    - [rotate-x, FRAC_PI_3]
- add: sphere
  transform:
  - rotation-x
  - [rotate-y, FRAC_PI_3]
"#;
        let (world, _) = parse(source);
        let transformation = Matrix::rotation_x(FRAC_PI_3)
            .rotate_y(FRAC_PI_3)
            .transformed();
        let sphere = Object::primitive(Shape::Sphere, Material::default(), transformation);
        assert_eq!(world.objects(), vec![sphere]);
    }

    #[test]
    fn merging_when_defining_add() {
        let source = r#"
- define: _sphere
  value:
    add: sphere
    material:
      color: [ 1, 0, 0 ]
      ambient: 0.5
      diffuse: 1
    transform:
    - [rotate-x, FRAC_PI_3]

- add: _sphere
  material:
    color: [ 0, 1, 0 ]
    ambient: 1
  transform:
    - [rotate-y, FRAC_PI_3]
"#;
        let (world, _) = parse(source);
        let material = Material {
            pattern: Pattern::Const(Color::green()),
            ambient: 1.,
            diffuse: 1.,
            ..Material::default()
        };
        let transformation = Matrix::rotation_x(FRAC_PI_3)
            .rotate_y(FRAC_PI_3)
            .transformed();
        let sphere = Object::primitive(Shape::Sphere, material, transformation);
        assert_eq!(world.objects(), vec![sphere]);
    }
}
