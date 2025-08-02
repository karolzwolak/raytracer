use std::{
    collections::HashMap,
    fmt::Display,
    path::{Path, PathBuf},
    str::FromStr,
};

use derive_builder::Builder;
use saphyr::Yaml;

use super::obj_model::ObjModelParser;
use crate::{
    math::{
        color::Color,
        matrix::Matrix,
        point::Point,
        transform::{
            Transformation, Transformations,
            local_transform::{LocalTransform, LocalTransformation, LocalTransformations},
        },
        tuple::{Axis, Tuple},
        vector::Vector,
    },
    scene::{
        ObjectGroup, PointLightSource, SceneBuilder,
        animation::{Animation, AnimationRepeat, Animations, TransformAnimation},
        camera::CameraBuilder,
        object::{
            Object, ObjectKind, PrimitiveObject,
            bounding_box::BoundingBox,
            csg::{CsgObject, CsgOperation},
            material::{Material, pattern::Pattern},
            primitive::{
                cone::Cone, cylinder::Cylinder, shape::Shape, smooth_triangle::SmoothTriangle,
                triangle::Triangle,
            },
        },
    },
};

#[derive(Debug, PartialEq, Builder)]
/// The output of the parser. It contains builders for scene and camera, because the cli options
/// can override the values in the yaml file.
/// It can also specify rendering options like quality options, animation duration and framerate.
pub struct YamlSceneConfig {
    #[builder(setter(strip_option), default = "None")]
    pub animation_duration_sec: Option<f64>,
    #[builder(setter(strip_option), default = "None")]
    pub animation_framerate: Option<u32>,

    #[builder(setter(strip_option), default = "None")]
    pub supersampling_level: Option<usize>,
    #[builder(setter(strip_option), default = "None")]
    pub max_reflective_depth: Option<usize>,

    #[builder(field(ty = "CameraBuilder", build = "self.camera_builder.clone()"))]
    pub camera_builder: CameraBuilder,
    #[builder(field(ty = "SceneBuilder", build = "self.scene_builder.clone()"))]
    pub scene_builder: SceneBuilder,
}

#[derive(Debug)]
pub enum YamlParseError {
    MissingField,
    InvalidField,
    UnexpectedValue,
    UnknownDefine(String),
    UnknownVariant(String),
    InvalidType(String),
    YamlSyntaxError(String),
    MultipleDocuments,
    FileReadError(String),
    ObjParsingError,
    InternalError,
    UnsupportedFeature(String),
    EmptyGroup, // empty groups don't make sense, it has to a mistake, so we return an error
}

impl YamlParseError {
    fn unsupported(feature: &str, _for: &str) -> Self {
        Self::UnsupportedFeature(format!(
            "The feature `{feature}` is unsupported for `{_for}`."
        ))
    }
}

impl Display for YamlParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

const PREDEFINED_DEFINES: &str = r#"
- define: PI
  value: 3.141592653589793
- define: 2_PI
  value: 6.283185307179586
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

- define: WHITE
  value: [ 1, 1, 1 ]
- define: BLACK
  value: [ 0, 0, 0 ]
- define: RED
  value: [ 1, 0, 0 ]
- define: GREEN
  value: [ 0, 1, 0 ]
- define: BLUE
  value: [ 0, 0, 1 ]

- define: GLASS_MATERIAL
  value:
    color: BLACK
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
    color: BLACK
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
    intensity: WHITE

- define: SCENE_CAMERA
  value:
    camera:
    from: [ 0, 1.5, -5 ]
    to: [ 0, 1, 0 ]
    up: [ 0, 1, 0 ]
    fov: 1.0471975511965979 # pi / 3
"#;

// TODO: Actual errors
pub struct YamlParser<'a> {
    yaml: &'a Yaml,
    input_path: Option<&'a Path>,
    result: YamlSceneConfigBuilder,
    defines: HashMap<String, Yaml>,
}

type YamlParseResult<T> = Result<T, YamlParseError>;
type YamlParserOutput = YamlParseResult<YamlSceneConfig>;

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
    ($values:ident, 2, $func:path) => {
        $func($values[0], $values[1])
    };
    ($values:ident, 3, $func:path) => {
        $func($values[0], $values[1], $values[2])
    };
    ($values:ident, 6, $func:path) => {
        $func(
            $values[0], $values[1], $values[2], $values[3], $values[4], $values[5],
        )
    };
}

macro_rules! parse_transformation {
    ($kind:ident, $values:ident, $n:tt) => {
        if $values.len() != $n {
            return Err(YamlParseError::InvalidField);
        } else {
            Ok(call_with_n_first_values!(
                $values,
                $n,
                Transformation::$kind
            ))
        }
    };
}

macro_rules! rotation_transformation {
    ($axis: expr, $values: ident) => {
        if $values.len() != 1 {
            return Err(YamlParseError::InvalidField);
        } else {
            Ok(Transformation::Rotation($axis, $values[0]))
        }
    };
}

impl<'a> YamlParser<'a> {
    fn file_read_error(&self, og_path: &str, path: &Path, err: std::io::Error) -> YamlParseError {
        let warning = if path.is_absolute() {
            ""
        } else {
            ".The path is relative, so it was resolved relative to the input file."
        };
        let scene_path = self.input_path.unwrap_or(Path::new(".")).to_string_lossy();
        let path = path.to_string_lossy();

        YamlParseError::FileReadError(format!(
            "Failed to read file `{path}`. Resolved from `{og_path}` specified in the scene `{scene_path}` : `{err}`{warning}"
        ))
    }
}

impl<'a> YamlParser<'a> {
    fn new(yaml: &'a Yaml, input_path: Option<&'a Path>, defines: HashMap<String, Yaml>) -> Self {
        Self {
            yaml,
            input_path,
            result: YamlSceneConfigBuilder::default(),
            defines,
        }
    }
    pub fn with_predefined_defines(yaml: &'a Yaml, input_path: Option<&'a Path>) -> Self {
        let parsed = saphyr::Yaml::load_from_str(PREDEFINED_DEFINES).unwrap();
        let predefined = parsed.first().unwrap();

        let mut predefined_parser = YamlParser::new(predefined, None, HashMap::new());
        predefined_parser
            .parse()
            .expect("Error parsing predefined defines");
        // println!("{:?}", predefined_parser.defines);

        YamlParser {
            yaml,
            result: YamlSceneConfigBuilder::default(),
            input_path,
            ..predefined_parser
        }
    }

    fn parse_num(&self, value: &Yaml) -> YamlParseResult<f64> {
        match value {
            Yaml::Integer(value) => Ok(*value as f64),
            Yaml::Real(value) => Ok(value.parse().unwrap()),
            Yaml::String(name) => {
                if let Some(string) = name.strip_prefix('-') {
                    let yaml = Yaml::from_str(string);
                    return self.parse_num(&yaml).map(|v| -v);
                }
                let value = self
                    .defines
                    .get(name)
                    .ok_or_else(|| YamlParseError::UnknownDefine(name.to_string()))?;
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
                .ok_or_else(|| YamlParseError::UnknownDefine(str_value.to_string()))?;
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
            val => Some(self.parse_matrix(val, "pattern")?),
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

    fn parse_camera(&mut self, body: &Yaml) -> YamlParseResult<()> {
        match &body["width"] {
            Yaml::BadValue => {}
            val => {
                self.result
                    .camera_builder
                    .target_width(self.parse_num(val)? as usize);
            }
        }
        match &body["height"] {
            Yaml::BadValue => {}
            val => {
                self.result
                    .camera_builder
                    .target_height(self.parse_num(val)? as usize);
            }
        }
        match &body["fov"] {
            Yaml::BadValue => {}
            val => {
                self.result
                    .camera_builder
                    .field_of_view(self.parse_num(val)?);
            }
        }
        match &body["field-of-view"] {
            Yaml::BadValue => {}
            val => {
                self.result
                    .camera_builder
                    .field_of_view(self.parse_num(val)?);
            }
        }

        if body["from"].is_badvalue() && body["to"].is_badvalue() && body["up"].is_badvalue() {
            return Ok(());
        }
        let from = self.parse_point(&body["from"])?;
        let to = self.parse_point(&body["to"])?;
        let up = self.parse_vector(&body["up"])?;

        let view = Matrix::view_tranformation(from, to, up);
        self.result.camera_builder.view_transformation(view);

        Ok(())
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
                    .ok_or_else(|| YamlParseError::UnknownDefine(name.to_string()))?;
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

    fn parse_singular_transformation_literal(
        &self,
        kind: &str,
        values: Vec<f64>,
    ) -> YamlParseResult<Transformation> {
        match kind {
            "translate" => parse_transformation!(Translation, values, 3),
            "scale" => parse_transformation!(Scaling, values, 3),
            "scale-uniform" => {
                let vector = if let Some(&val) = values.first() {
                    vec![val, val, val]
                } else {
                    values
                };
                parse_transformation!(Scaling, vector, 3)
            }
            "rotate-x" => rotation_transformation!(Axis::X, values),
            "rotate-y" => rotation_transformation!(Axis::Y, values),
            "rotate-z" => rotation_transformation!(Axis::Z, values),
            "shear" => parse_transformation!(Shearing, values, 6),
            _ => Err(YamlParseError::InvalidField),
        }
    }

    fn extract_transformation_literal(
        &'a self,
        body: &'a Yaml,
    ) -> YamlParseResult<(&'a str, Vec<f64>)> {
        let values = body.as_vec().ok_or(YamlParseError::InvalidField)?;
        if values.is_empty() {
            return Err(YamlParseError::MissingField);
        }
        let kind = values[0].as_str().ok_or(YamlParseError::InvalidField)?;
        let values = self.parse_vec(&values[1..])?;

        Ok((kind, values))
    }

    fn parse_singular_local_transformation(
        &self,
        body: &Yaml,
    ) -> YamlParseResult<LocalTransformation> {
        let (kind, values) = self.extract_transformation_literal(body)?;

        if let Some(kind) = kind.strip_prefix("local-") {
            return self
                .parse_singular_transformation_literal(kind, values)
                .map(LocalTransformation::Local);
        }

        Ok(match kind {
            "center" => LocalTransformation::Center,
            "normalize-all-axes" => LocalTransformation::NormalizeAllAxes,
            "normalize-to-longest-axis" => LocalTransformation::NormalizeToLongestAxis,

            "translate-above-x" => LocalTransformation::TranslateAbove(Axis::X),
            "translate-above-y" => LocalTransformation::TranslateAbove(Axis::Y),
            "translate-above-z" => LocalTransformation::TranslateAbove(Axis::Z),

            "translate-below-x" => LocalTransformation::TranslateBelow(Axis::X),
            "translate-below-y" => LocalTransformation::TranslateBelow(Axis::Y),
            "translate-below-z" => LocalTransformation::TranslateBelow(Axis::Z),

            _ => {
                return self
                    .parse_singular_transformation_literal(kind, values)
                    .map(LocalTransformation::Transformation);
            }
        })
    }

    fn parse_matrix(&self, body: &Yaml, _for: &str) -> YamlParseResult<Matrix> {
        let transformations = Transformations::try_from(self.parse_transformations(body)?)
            .map_err(|_| YamlParseError::unsupported("local transformations", _for))?;
        Ok(Matrix::from(&transformations))
    }

    fn parse_transformations(&self, body: &Yaml) -> YamlParseResult<LocalTransformations> {
        match body {
            Yaml::BadValue => Ok(LocalTransformations::new()),
            Yaml::String(name) => {
                let transform = self
                    .defines
                    .get(name)
                    .ok_or_else(|| YamlParseError::UnknownDefine(name.to_string()))?;
                self.parse_transformations(transform)
            }
            Yaml::Array(arr) => {
                let mut res = LocalTransformations::new();
                for val in arr {
                    match val {
                        Yaml::String(name) => {
                            let transform = self
                                .defines
                                .get(name)
                                .ok_or_else(|| YamlParseError::UnknownDefine(name.to_string()))?;
                            res.extend(&self.parse_transformations(transform)?);
                        }
                        _ => res.push(self.parse_singular_local_transformation(val)?),
                    }
                }
                Ok(res)
            }
            _ => Err(YamlParseError::InvalidField),
        }
    }

    fn parse_group(&self, body: &Yaml) -> YamlParseResult<ObjectGroup> {
        let children_yaml = body["children"]
            .as_vec()
            .ok_or(YamlParseError::EmptyGroup)?;
        if children_yaml.is_empty() {
            return Err(YamlParseError::EmptyGroup);
        }
        children_yaml
            .iter()
            .map(|yaml| self.parse_object(yaml))
            .collect::<YamlParseResult<Vec<Object>>>()
            .map(ObjectGroup::new)
    }

    fn resolve_path_from_scene(&self, path: &Path) -> PathBuf {
        let input_dir = match self.input_path {
            Some(input_path) => input_path.parent().unwrap_or_else(|| Path::new(".")),
            None => Path::new("."),
        };

        input_dir.join(PathBuf::from(path))
    }

    fn parse_obj_model(&self, body: &Yaml) -> YamlParseResult<ObjectGroup> {
        let file_path = body["file"].as_str().ok_or(YamlParseError::MissingField)?;
        let path = self.resolve_path_from_scene(&PathBuf::from(file_path));

        let data = std::fs::read_to_string(&path)
            .map_err(|err| self.file_read_error(file_path, &path, err))?;

        let parser = ObjModelParser::new();
        parser
            .parse(data)
            .map_err(|_| YamlParseError::ObjParsingError)
    }

    fn parse_str_or_default<T>(yaml: &Yaml, key: &str) -> YamlParseResult<T>
    where
        T: FromStr + Default,
    {
        match &yaml[key] {
            &Yaml::BadValue => Ok(Default::default()),
            Yaml::String(val) => val
                .parse()
                .map_err(|_| YamlParseError::UnknownVariant(val.to_owned())),
            _ => Err(YamlParseError::InvalidType(format!(
                "{key} must be a string"
            )))?,
        }
    }

    fn parse_animation(&self, body: &Yaml) -> YamlParseResult<TransformAnimation> {
        let transformations = self.parse_transformations(&body["transform"])?;
        let duration = self.parse_num(&body["duration"])?;
        let delay = match &body["delay"] {
            &Yaml::BadValue => 0.,
            val => self.parse_num(val)?,
        };

        let direction = Self::parse_str_or_default(body, "direction")?;
        let timing = Self::parse_str_or_default(body, "timing")?;

        let count = match &body["repeat"] {
            &Yaml::BadValue => Default::default(),
            &Yaml::Integer(val) if val >= 0 => AnimationRepeat::Repeat(val as u32),
            Yaml::String(name) if name == "infinite" => AnimationRepeat::Infinite,
            _ => return Err(YamlParseError::InvalidField),
        };

        let anim = Animation::new(delay, duration, direction, timing, count);
        Ok(TransformAnimation::new(anim, transformations))
    }

    fn parse_animations(&self, body: &Yaml) -> YamlParseResult<Animations> {
        let animations = body.as_vec().ok_or(YamlParseError::InvalidField)?;
        let vec = animations
            .iter()
            .map(|yaml| self.parse_animation(yaml))
            .collect::<YamlParseResult<Vec<TransformAnimation>>>()?;
        Ok(Animations::with_vec(vec))
    }

    fn parse_object(&self, body: &Yaml) -> YamlParseResult<Object> {
        let kind = body["add"].as_str().ok_or(YamlParseError::InvalidField)?;
        self.parse_object_with_kind(body, kind)
    }

    // Option because the bbox is optional, so None means don't add bbox
    fn parse_object_debug_bbox_material(&self, body: &Yaml) -> YamlParseResult<Option<Material>> {
        if body.is_badvalue() {
            return Ok(None);
        }

        match &body["material"] {
            Yaml::BadValue => Ok(Some(BoundingBox::DEFAULT_DEBUG_BBOX_MATERIAL)),
            body => self.parse_material(body).map(Some),
        }
    }

    fn parse_csg_object(&self, body: &Yaml) -> YamlParseResult<CsgObject> {
        let left = self.parse_object(&body["left"])?;
        let right = self.parse_object(&body["right"])?;
        let operation = body["operation"]
            .as_str()
            .ok_or(YamlParseError::InvalidField)?
            .parse::<CsgOperation>()
            .map_err(|_| {
                YamlParseError::UnknownVariant(body["operation"].as_str().unwrap().to_string())
            })
            .unwrap();
        Ok(CsgObject::new(operation, left, right))
    }

    fn parse_object_with_kind(&self, body: &Yaml, obj_kind: &str) -> YamlParseResult<Object> {
        let animations = match &body["animate"] {
            Yaml::BadValue => Animations::empty(),
            val => self.parse_animations(val)?,
        };
        let material = self.parse_material(&body["material"])?;
        let transformations = self.parse_transformations(&body["transform"])?;
        let bbox_material = self.parse_object_debug_bbox_material(&body["bbox"])?;

        let shape = match obj_kind {
            "group" | "obj" | "csg" => {
                let kind = match obj_kind {
                    "group" => self.parse_group(body).map(ObjectKind::Group),
                    "obj" => self.parse_obj_model(body).map(ObjectKind::Group),
                    "csg" => self
                        .parse_csg_object(body)
                        .map(Box::new)
                        .map(ObjectKind::Csg),
                    _ => unreachable!(),
                }?;
                let mut res = Object::animated(kind, animations);
                if material != Material::default() {
                    res.set_material(material);
                }
                if !transformations.vec().is_empty() {
                    res.local_transform(&transformations);
                }

                if let Some(material) = bbox_material {
                    res.into_group_with_bbox(material);
                }
                return Ok(res);
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
                let y_min = self.parse_num(&body["min"])?;
                let y_max = self.parse_num(&body["max"])?;
                let closed = self.parse_bool(&body["closed"])?;
                Shape::Cone(Cone {
                    y_min,
                    y_max,
                    closed,
                })
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
                    return self.parse_object_with_kind(&body, what);
                }
                return Err(YamlParseError::InvalidField);
            }
        };
        let mut obj = Object::animated(
            ObjectKind::primitive(PrimitiveObject::with_shape_material(shape, material)),
            animations,
        );
        if !transformations.vec().is_empty() {
            obj.local_transform(&transformations);
        }

        if let Some(material) = bbox_material {
            obj.into_group_with_bbox(material);
        }
        Ok(obj)
    }

    fn parse_options(&mut self, body: &Yaml) -> YamlParseResult<()> {
        match &body["supersampling-level"] {
            Yaml::BadValue => {}
            val => {
                self.result
                    .supersampling_level(self.parse_num(val)? as usize);
            }
        }
        match &body["reflective-depth"] {
            Yaml::BadValue => {}
            val => {
                self.result
                    .max_reflective_depth(self.parse_num(val)? as usize);
            }
        }
        Ok(())
    }

    fn parse_add(&mut self, what: &str, body: &Yaml) -> YamlParseResult<()> {
        match what {
            "camera" => self.parse_camera(body)?,
            "light" => {
                let light = self.parse_light(body)?;
                self.result.scene_builder.add_light_source(light);
            }
            "group" | "obj" | "sphere" | "cube" | "plane" | "cylinder" | "cone" | "triangle"
            | "smooth-triangle" | "csg" => {
                let object = self.parse_object_with_kind(body, what)?;
                self.result.scene_builder.add_object(object);
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
            (Yaml::Hash(define_hash), Yaml::Hash(use_hash)) => {
                return self.merge_hash(name, define_hash, use_hash);
            }
            (Yaml::Array(define_array), Yaml::Array(use_array)) if key_str == "transform" => {
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
                    "camera" => self.parse_camera(yaml_obj)?,
                    "options" => self.parse_options(yaml_obj)?,
                    _ => {}
                },
                Some((Yaml::String(operation), Yaml::Null)) => match operation.as_str() {
                    "camera" => self.parse_camera(yaml_obj)?,
                    "options" => self.parse_options(yaml_obj)?,
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

        Ok(self.result.build().unwrap())
    }
}

impl YamlParser<'_> {
    fn str_to_yaml(source: &str) -> YamlParseResult<Yaml> {
        let mut docs = saphyr::Yaml::load_from_str(source)
            .map_err(|_| YamlParseError::YamlSyntaxError(source.to_string()))?;
        match docs.len() {
            1 => Ok(std::mem::replace(&mut docs[0], Yaml::BadValue)),
            0 => Ok(Yaml::Array(vec![])),
            _ => Err(YamlParseError::MultipleDocuments),
        }
    }
}

fn parse(source: &str, input_path: Option<&Path>) -> YamlParserOutput {
    let yaml = YamlParser::str_to_yaml(source)?;
    let parser = YamlParser::with_predefined_defines(&yaml, input_path);

    parser.parse_consume()
}

pub fn parse_file(source: &str, input_path: &Path) -> YamlParserOutput {
    parse(source, Some(input_path))
}

#[cfg(test)]
mod tests {
    const SAMPLE_SCENES_DIRS: [&str; 3] = [
        "samples/chapters/",
        "samples/scenes/",
        "samples/animations/",
    ];

    use std::{f64::consts::FRAC_PI_3, fmt::Debug, path::PathBuf};

    use super::*;
    use crate::{
        math::transform::Transform,
        scene::{
            Scene,
            animation::{
                Animation, AnimationDirection, AnimationRepeat, AnimationTiming, TransformAnimation,
            },
            camera::Camera,
        },
    };

    const WIDTH: usize = 600;
    const HEIGHT: usize = 800;
    const FOV: f64 = std::f64::consts::PI / 2.0;

    fn parse_str(source: &str) -> YamlParserOutput {
        parse(source, None)
    }

    fn test_parse(source: &str) -> (Scene, Camera) {
        let mut config = parse_str(source).unwrap();
        config
            .camera_builder
            .default_target_width(WIDTH)
            .default_target_height(HEIGHT)
            .default_field_of_view(FOV);

        (
            config.scene_builder.build(),
            config.camera_builder.build().unwrap(),
        )
    }

    fn test_parse_object<T, S: ToString, F>(
        source: &str,
        test_values: Vec<S>,
        expected: Vec<T>,
        getter: F,
    ) where
        T: PartialEq + Debug,
        F: Fn(&Object) -> T,
    {
        let source = test_values
            .into_iter()
            .map(|s| source.replace("{}", &s.to_string()))
            .collect::<String>();
        let (scene, _) = test_parse(&source);
        let actual = scene
            .objects()
            .children()
            .iter()
            .map(getter)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    fn test_field<T, F>(
        source: &str,
        field: &str,
        test_values: Vec<&str>,
        expected: Vec<T>,
        field_getter: F,
    ) where
        T: PartialEq + Debug,
        F: Fn(&Object) -> T,
    {
        let test_values = test_values
            .into_iter()
            .map(|s| {
                if s.is_empty() {
                    String::new()
                } else {
                    format!("{field}: {s}")
                }
            })
            .collect::<Vec<String>>();
        test_parse_object(source, test_values, expected, field_getter)
    }

    #[test]
    fn empty_yaml() {
        let (scene, camera) = test_parse("");
        assert_eq!(scene, Scene::default());
        assert_eq!(camera, Camera::new(WIDTH, HEIGHT, FOV));
    }
    #[test]
    fn comments_are_supported() {
        let source = "#comment";
        let _ = test_parse(source);
    }

    #[test]
    fn parse_light() {
        const LIGHT_YAML: &str = r#"
- add: light
  at: [ 50, 100, -50 ]
  intensity: [ 1, 1, 1]
"#;
        let (scene, _) = test_parse(LIGHT_YAML);
        let expected_light = PointLightSource::new(Point::new(50., 100., -50.), Color::white());
        assert_eq!(scene.light_sources().first(), Some(&expected_light));
    }

    #[test]
    fn parse_camera() {
        const CAMERA_YAML: &str = r#"
- camera:
  width: 100
  height: 100
  field-of-view: 0.785
  from: [ -6, 6, -10 ]
  to: [ 6, 0, 6 ]
  up: [ -0.45, 1, 0 ]
"#;
        let (_, camera) = test_parse(CAMERA_YAML);
        let view = Matrix::view_tranformation(
            Point::new(-6., 6., -10.),
            Point::new(6., 0., 6.),
            Vector::new(-0.45, 1., 0.),
        );
        let expected_camera = Camera::with_transformation(100, 100, 0.785, view);
        assert_eq!(camera, expected_camera);
    }

    #[test]
    fn camera_without_from_up_to() {
        const CAMERA_YAML: &str = r#"
- camera:
  width: 100
  height: 100
  field-of-view: 0.785
"#;
        let (_, camera) = test_parse(CAMERA_YAML);
        let expected_camera = Camera::new(100, 100, 0.785);
        assert_eq!(camera, expected_camera);
    }

    #[test]
    fn add_camera() {
        const CAMERA_YAML: &str = r#"
- add: camera
  from: [0, 2.5, -10]
  to: [0, 1, 0]
  up: [0, 1, 0]
  "#;
        let (_, camera) = test_parse(CAMERA_YAML);
        let expected_camera = Camera::with_transformation(
            WIDTH,
            HEIGHT,
            FOV,
            Matrix::view_tranformation(
                Point::new(0., 2.5, -10.),
                Point::new(0., 1., 0.),
                Vector::new(0., 1., 0.),
            ),
        );
        assert_eq!(camera, expected_camera);
    }

    #[test]
    fn parse_options() {
        const SCENE_YAML: &str = r#"
- options:
  reflective-depth: 4
  supersampling-level: 3
"#;
        let config = parse_str(SCENE_YAML).unwrap();
        assert_eq!(config.max_reflective_depth, Some(4));
        assert_eq!(config.supersampling_level, Some(3));
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
        let (scene, _) = test_parse(PLANE_YAML);
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

        assert_eq!(scene.objects().children(), vec![expected_object]);
    }

    #[test]
    fn objects_have_default_material_and_transformation() {
        const DEFAULT_SPHERE_YAML: &str = r#"
- add: sphere
"#;
        let (scene, _) = test_parse(DEFAULT_SPHERE_YAML);
        let sphere = Object::primitive(Shape::Sphere, Material::default(), Matrix::identity());
        assert_eq!(scene.objects().children(), vec![sphere]);
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
        let (scene, _) = test_parse(SPHERE_YAML);
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

        assert_eq!(scene.objects().children(), vec![expected_object]);
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
        let (scene, _) = test_parse(DEFINE_MATERIALS_YAML);
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

        assert_eq!(scene.objects().children(), expected_objects);
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
        let (scene, _) = test_parse(DEFINE_TRANSFORMS_YAML);
        let standard_transform = Matrix::translation(1., -1., 1.)
            .scale(0.5, 0.5, 0.5)
            .transformed();
        let large_object_transform = standard_transform.clone().scale_uniform(4.).transformed();
        let sphere = Object::primitive(Shape::Sphere, Material::default(), standard_transform);
        let cube = Object::primitive(Shape::Cube, Material::default(), large_object_transform);
        let expected_objects = vec![sphere, cube];
        assert_eq!(scene.objects().children(), expected_objects);
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
        let (scene, _) = test_parse(GROUP_YAML);
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

        assert_eq!(scene.objects().children(), vec![Object::from_group(group)]);
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
            let res = parse_str(source);
            assert!(matches!(res, Err(YamlParseError::EmptyGroup)));
        }
    }

    #[test]
    fn parse_obj() {
        const OBJ_YAML: &str = r#"
- add: obj
  file: samples/obj/teapot-low.obj
"#;
        let (scene, _) = test_parse(OBJ_YAML);
        let parser = ObjModelParser::new();
        let path = "samples/obj/teapot-low.obj";
        let data = std::fs::read_to_string(path).unwrap();
        let expected_group = parser.parse(data).unwrap();
        assert_eq!(
            scene.objects().children(),
            vec![Object::from_group(expected_group)]
        );
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
        let (scene, _) = test_parse(USE_DEFINE_IN_ADD_YAML);
        let red_material = Material {
            pattern: Pattern::Const(Color::red()),
            ..Material::default()
        };
        let red_cube = Object::primitive(Shape::Cube, red_material, Matrix::identity());
        let expected_objects = vec![red_cube];
        assert_eq!(scene.objects().children(), expected_objects);
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

        let (scene, _) = test_parse(DEFINE_COLOR_YAML);
        let red_material = Material {
            pattern: Pattern::Const(Color::red()),
            ..Material::default()
        };
        let red_sphere = Object::primitive(Shape::Sphere, red_material, Matrix::identity());
        let expected_objects = vec![red_sphere];
        assert_eq!(scene.objects().children(), expected_objects);
    }

    #[test]
    fn parse_cylinder() {
        const CYLINDER_YAML: &str = r#"
- add: cylinder
  min: 1
  max: 5
  closed: true
"#;

        let (scene, _) = test_parse(CYLINDER_YAML);
        let cylinder_shape = Cylinder::new(1., 5., true);
        let expected_object = Object::primitive_with_shape(Shape::Cylinder(cylinder_shape));
        assert_eq!(scene.objects().children(), vec![expected_object]);
    }

    #[test]
    fn parse_cone() {
        const CONE_YAML: &str = r#"
- add: cone
  min: 1
  max: 5
  closed: true
"#;

        let (scene, _) = test_parse(CONE_YAML);
        let cylinder_shape = Cone {
            y_min: 1.,
            y_max: 5.,
            closed: true,
        };
        let expected_object = Object::primitive_with_shape(Shape::Cone(cylinder_shape));
        assert_eq!(scene.objects().children(), vec![expected_object]);
    }

    #[test]
    fn parse_triangle() {
        const TRIANGLE_YAML: &str = r#"
- add: triangle
  p1: [ 0, 1, 0 ]
  p2: [ -1, 0, 0 ]
  p3: [ 1, 0, 0 ]
"#;

        let (scene, _) = test_parse(TRIANGLE_YAML);
        let triangle = Object::primitive_with_shape(Shape::Triangle(Triangle::new(
            Point::new(0., 1., 0.),
            Point::new(-1., 0., 0.),
            Point::new(1., 0., 0.),
        )));
        let expected_objects = vec![triangle];
        assert_eq!(scene.objects().children(), expected_objects);
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

        let (scene, _) = test_parse(SMOOTH_TRIANGLE_YAML);
        let triangle = Object::primitive_with_shape(Shape::SmoothTriangle(SmoothTriangle::new(
            Point::new(0., 1., 0.),
            Point::new(-1., 0., 0.),
            Point::new(1., 0., 0.),
            Vector::new(0., 1., 0.),
            Vector::new(-1., 0., 0.),
            Vector::new(1., 0., 0.),
        )));
        let expected_objects = vec![triangle];
        assert_eq!(scene.objects().children(), expected_objects);
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
        let (scene, _) = test_parse(PATTERNS_YAML);
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
        assert_eq!(scene.objects().children(), expected_objects);
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
        let (scene, _) = test_parse(source);
        let glass_sphere = Object::primitive(Shape::Sphere, Material::glass(), Matrix::identity());
        let mirror_sphere =
            Object::primitive(Shape::Sphere, Material::mirror(), Matrix::identity());
        let air_sphere = Object::primitive(Shape::Sphere, Material::air(), Matrix::identity());
        let expected_objects = vec![glass_sphere, mirror_sphere, air_sphere];
        assert_eq!(scene.objects().children(), expected_objects);
    }

    #[test]
    fn predefined_scene_light_and_camera() {
        let source = r#"
- add: SCENE_LIGHT
- add: SCENE_CAMERA
"#;
        let (scene, camera) = test_parse(source);
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
        assert_eq!(scene.light_sources().first(), Some(&light));
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
    refractive-index: 2_PI
"#;
        let (scene, _) = test_parse(source);
        let material = Material {
            refractive_index: 2. * std::f64::consts::PI,
            ambient: std::f64::consts::PI,
            diffuse: std::f64::consts::FRAC_PI_2,
            specular: std::f64::consts::FRAC_PI_3,
            shininess: std::f64::consts::FRAC_PI_4,
            reflectivity: std::f64::consts::FRAC_PI_6,
            transparency: std::f64::consts::FRAC_1_SQRT_2,
            ..Material::default()
        };
        let sphere = Object::primitive(Shape::Sphere, material, Matrix::identity());
        assert_eq!(scene.objects().children(), vec![sphere]);
    }

    #[test]
    fn predefined_colors() {
        let source = r#"
- add: sphere
  material:
    color: WHITE
- add: sphere
  material:
    color: BLACK
- add: sphere
  material:
    color: RED
- add: sphere
  material:
    color: GREEN
- add: sphere
  material:
    color: BLUE
"#;
        let (scene, _) = test_parse(source);
        let colors = vec![
            Color::white(),
            Color::black(),
            Color::red(),
            Color::green(),
            Color::blue(),
        ];
        let actual_colors = scene
            .objects()
            .children()
            .iter()
            .map(|obj| match obj.material().unwrap().pattern {
                Pattern::Const(color) => color,
                _ => panic!("Expected a constant color pattern"),
            })
            .collect::<Vec<_>>();
        assert_eq!(colors, actual_colors);
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
        let (scene, _) = test_parse(source);
        let green_material = Material {
            pattern: Pattern::Const(Color::green()),
            ambient: 1.,
            diffuse: 1.,
            ..Material::default()
        };
        let sphere = Object::primitive(Shape::Sphere, green_material, Matrix::identity());
        assert_eq!(scene.objects().children(), vec![sphere]);
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
        let (scene, _) = test_parse(source);
        let transformation = Matrix::rotation_x(FRAC_PI_3)
            .rotate_y(FRAC_PI_3)
            .transformed();
        let sphere = Object::primitive(Shape::Sphere, Material::default(), transformation);
        assert_eq!(scene.objects().children(), vec![sphere]);
    }

    #[test]
    fn define_transformation() {
        let source = r#"
- define: fancy-transform
  value:
    - [rotate-x, FRAC_PI_3]
    - [scale, 1, 2, 3]
- add: sphere
  transform:
  - [scale-uniform, 2]
  - fancy-transform
  - [translate, 1, 2, 3]
- add: cube
  transform: fancy-transform
"#;
        let (scene, _) = test_parse(source);
        let fancy_transformation = Matrix::rotation_x(FRAC_PI_3)
            .scale(1., 2., 3.)
            .transformed();
        let transformations = vec![
            fancy_transformation
                .clone()
                .scale_uniform(2.)
                .translate(1., 2., 3.)
                .transformed(),
            fancy_transformation,
        ];
        let actual_transformations = scene
            .objects()
            .children()
            .iter()
            .map(|obj| obj.transformation())
            .collect::<Vec<_>>();

        assert_eq!(transformations, actual_transformations);
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
        let (scene, _) = test_parse(source);
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
        assert_eq!(scene.objects().children(), vec![sphere]);
    }

    #[test]
    fn parsing_negative_nums() {
        let source = r#"
- add: sphere
  transform:
    - [translate, -1, -FRAC_PI_2, -5.5]
"#;
        let (scene, _) = test_parse(source);
        let sphere = Object::primitive(
            Shape::Sphere,
            Material::default(),
            Matrix::translation(-1., -std::f64::consts::FRAC_PI_2, -5.5).transformed(),
        );
        assert_eq!(scene.objects().children(), vec![sphere]);
    }

    #[test]
    fn object_animations() {
        let source = r#"
- add: sphere
  animate:
    - delay: 2
      duration: 5
      direction: reverse
      timing: linear
      repeat: 1
      transform:
        - [translate, 1, -2, 10]
        - [rotate-y, -FRAC_PI_3]
"#;
        let (scene, _) = test_parse(source);
        let animations = Animations::with_vec(vec![TransformAnimation::new(
            Animation::new(
                2.,
                5.,
                AnimationDirection::Reverse,
                AnimationTiming::Linear,
                AnimationRepeat::Repeat(1),
            ),
            Transformations::from(vec![
                Transformation::Translation(1., -2., 10.),
                Transformation::Rotation(Axis::Y, -std::f64::consts::FRAC_PI_3),
            ])
            .into(),
        )]);

        assert_eq!(
            scene.objects().children().first().unwrap().animations(),
            &animations
        );
    }
    #[test]
    fn parse_animation_timing() {
        let source = r#"
- add: sphere
  animate:
    - delay: 0
      duration: 1
      {}
"#;
        let timings_str = vec!["linear", "ease-in", "ease-out", "ease-in-out", ""];
        let expected_timings = vec![
            AnimationTiming::Linear,
            AnimationTiming::EaseIn,
            AnimationTiming::EaseOut,
            AnimationTiming::EaseInOut,
            AnimationTiming::default(),
        ];
        test_field(source, "timing", timings_str, expected_timings, |obj| {
            obj.animations().vec().first().unwrap().animations().timing
        });
    }
    #[test]
    fn parse_animation_direction() {
        let source = r#"
- add: sphere
  animate:
    - delay: 0
      duration: 1
      {}
"#;
        let timings_str = vec!["normal", "reverse", "alternate", "alternate-reverse", ""];
        let expected_timings = vec![
            AnimationDirection::Normal,
            AnimationDirection::Reverse,
            AnimationDirection::Alternate,
            AnimationDirection::AlternateReverse,
            AnimationDirection::default(),
        ];
        test_field(source, "direction", timings_str, expected_timings, |obj| {
            obj.animations()
                .vec()
                .first()
                .unwrap()
                .animations()
                .direction
        });
    }
    #[test]
    fn parse_csg() {
        let source = r#"
- add: csg
  operation: union
  left:
    add: sphere
  right:
    add: cube
    transform:
      - [translate, 1, 2, 3]
"#;
        let kind = ObjectKind::Csg(Box::new(CsgObject::new(
            CsgOperation::Union,
            Object::primitive_with_shape(Shape::Sphere),
            Object::primitive_with_transformation(Shape::Cube, Matrix::translation(1., 2., 3.)),
        )));
        let expected_object = Object::animated(kind, Animations::default());
        let (scene, _) = test_parse(source);
        assert_eq!(scene.objects().children(), vec![expected_object]);
    }

    #[test]
    fn parse_csg_operations() {
        let source = r#"
- add: csg
  {}
  left:
    add: sphere
  right:
    add: cube
"#;
        test_field(
            source,
            "operation",
            vec!["union", "intersection", "difference"],
            vec![
                CsgOperation::Union,
                CsgOperation::Intersection,
                CsgOperation::Difference,
            ],
            |obj| match obj.kind() {
                ObjectKind::Csg(csg) => csg.operation,
                _ => panic!("Expected CSG object"),
            },
        );
    }

    #[test]
    fn parse_local_transformations() {}

    #[test]
    fn add_debug_bbox_to_obj() {
        let source = r#"
- add: cube
  bbox:
    {}
        "#;
        let material_strings = vec![
            r#"
    material:
        color: RED
"#,
            "",
        ];

        let expected_materials = vec![
            Material::with_color(Color::red()),
            BoundingBox::DEFAULT_DEBUG_BBOX_MATERIAL,
        ];

        test_parse_object(source, material_strings, expected_materials, |obj| {
            obj.as_group()
                .unwrap()
                .children()
                .iter()
                .find(|obj| {
                    obj.as_primitive()
                        .is_some_and(|p| matches!(p.shape(), Shape::Bbox))
                })
                .unwrap()
                .material_unwrapped()
                .clone()
        });
    }

    #[test]
    fn bad_file_in_scene() {
        let source = r#"
- add: obj
  file: {}
    "#;
        let non_existent_file = "non_existent_file.obj";
        assert!(
            !PathBuf::from(non_existent_file).exists(),
            "Test file should not exist"
        );
        let source = source.replace("{}", non_existent_file);
        let res = parse_str(&source);
        assert!(
            matches!(res, Err(YamlParseError::FileReadError(_)),),
            "{res:?}"
        );
    }

    #[test]
    fn files_in_scenes_are_resolved_relative_to_input_path() {
        let file = PathBuf::from("some/path/to/file/model.obj");
        let input_path = PathBuf::from("input/path/to/scene.yml");

        let parser = YamlParser::new(&Yaml::Null, Some(&input_path), HashMap::new());

        let resolved_path = parser.resolve_path_from_scene(&file);
        assert_eq!(
            resolved_path,
            PathBuf::from("input/path/to/some/path/to/file/model.obj")
        );
    }

    fn get_sample_scenes() -> Vec<PathBuf> {
        SAMPLE_SCENES_DIRS
            .iter()
            .flat_map(|dir| {
                PathBuf::from(dir)
                    .read_dir()
                    .unwrap()
                    .map(|entry| entry.unwrap().path())
                    .filter(|entry| {
                        entry
                            .extension()
                            .is_some_and(|ext| ext == "yaml" || ext == "yml")
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    #[ignore]
    fn parse_sample_scenes() {
        use rayon::prelude::*;

        let scenes = get_sample_scenes();
        assert!(!scenes.is_empty(), "No sample scenes found");

        scenes.par_iter().for_each(|scene| {
            let now = std::time::Instant::now();
            let path = scene.to_str().unwrap();
            let source = std::fs::read_to_string(path).unwrap();
            let _ = parse_str(&source).unwrap_or_else(|_| panic!("Failed to parse {scene:?}"));
            println!("Parsed {:?} in {:?}", scene, now.elapsed());
        });
    }
}
