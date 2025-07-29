pub mod animation;
pub mod camera;
pub mod io;
pub mod light;
pub mod object;

use light::point_light::PointLightSource;
use object::group::ObjectGroup;

use crate::scene::object::Object;

#[derive(PartialEq, Debug, Clone, Default)]
/// Scene containing objects and light sources, can be animated and compute BVH.
pub struct Scene {
    /// The objects in the scene
    objects: ObjectGroup,
    /// The light sources in the scene
    light_sources: Vec<PointLightSource>,
}

impl Scene {
    pub fn objects(&self) -> &ObjectGroup {
        &self.objects
    }

    pub fn light_sources(&self) -> &[PointLightSource] {
        &self.light_sources
    }

    pub fn animate(&self, time: f64) -> Self {
        let mut clone = self.clone();
        clone.objects.animate(time);
        clone.compute_bvh();

        clone
    }

    pub fn compute_bvh(&mut self) {
        self.objects.build_bvh();
    }
}

#[cfg(test)]
impl Scene {
    #[cfg(test)]
    pub fn default_testing() -> Self {
        SceneBuilder::default_builder_testing().build()
    }
}

/// The builder for the scene.
#[derive(PartialEq, Debug, Clone, Default)]
pub struct SceneBuilder {
    /// The objects in the scene
    objects: ObjectGroup,
    /// The light sources in the scene
    light_sources: Vec<PointLightSource>,
}

#[cfg(test)]
impl SceneBuilder {
    #[cfg(test)]
    // Default testing scene without bvh
    pub fn default_builder_testing() -> Self {
        use object::{
            material::{Material, pattern::Pattern},
            primitive::shape::Shape,
        };

        use crate::{
            math::{color::Color, matrix::Matrix, point::Point, tuple::Tuple},
            scene::object::PrimitiveObject,
        };

        let sphere1 = Object::primitive(
            Shape::Sphere,
            Material {
                pattern: Pattern::Const(Color::new(0.8, 1.0, 0.6)),
                ambient: 0.1,
                diffuse: 0.7,
                specular: 0.2,
                ..Default::default()
            },
            Matrix::identity(),
        );
        let sphere2 =
            PrimitiveObject::with_transformation(Shape::Sphere, Matrix::scaling(0.5, 0.5, 0.5))
                .into();

        let objects = vec![sphere1, sphere2];
        let lights = vec![PointLightSource::new(
            Point::new(-10., 10., -10.),
            Color::white(),
        )];

        let mut res = SceneBuilder::default();
        res.objects(objects).light_sources(lights);

        res
    }
}

impl SceneBuilder {
    pub fn add_light_source(&mut self, light_source: PointLightSource) -> &mut Self {
        self.light_sources.push(light_source);
        self
    }

    pub fn light_sources(&mut self, light_sources: Vec<PointLightSource>) -> &mut Self {
        self.light_sources = light_sources;
        self
    }

    pub fn objects<T>(&mut self, objects: T) -> &mut Self
    where
        T: Into<ObjectGroup>,
    {
        self.objects = objects.into();
        self
    }

    pub fn get_objects(&self) -> &ObjectGroup {
        &self.objects
    }

    pub fn get_objects_mut(&mut self) -> &mut ObjectGroup {
        &mut self.objects
    }

    pub fn add_object(&mut self, object: Object) -> &mut Self {
        self.objects.add_child(object);
        self
    }
}

impl SceneBuilder {
    pub fn build(self) -> Scene {
        Scene {
            objects: self.objects,
            light_sources: self.light_sources,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{object::primitive::shape::Shape, *};

    fn animated_builder() -> SceneBuilder {
        let mut builder = SceneBuilder::default();
        builder.add_light_source(PointLightSource::default());
        builder.add_object(Object::primitive_with_shape(Shape::Sphere));
        builder.add_object(Object::primitive_with_shape(Shape::Cube));
        builder.add_object(Object::primitive_with_shape(Shape::Plane));
        builder.add_object(Object::animated_testing());
        builder
    }

    #[test]
    fn animating_scene() {
        let time = 0.5;
        let scene = animated_builder().build();

        let mut wrong = scene.clone();
        wrong.compute_bvh();
        wrong.objects.animate(time);

        let mut expected = scene.clone();
        expected.objects.animate(time);
        expected.compute_bvh();

        let scene = scene.animate(time);

        assert_ne!(
            expected, wrong,
            "The test logic is broken, expected == wrong; {expected:?}"
        );

        assert_eq!(scene, expected);
        assert_ne!(scene, wrong);
    }
}
