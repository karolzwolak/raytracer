use crate::{
    math::{matrix::Matrix, point::Point, tuple::Tuple},
    render::{image::Image, ray::Ray},
};

// camera looks toward -z direction from point zero
// this makes +x to be on the left
#[derive(PartialEq, Debug, Clone, Default)]
pub struct Camera {
    target_width: usize,
    target_height: usize,
    inverse_transformation: Matrix,

    pixel_size: f64,
    half_width: f64,
    half_height: f64,
    field_of_view: f64,
}

pub type CameraBuilderError = derive_builder::UninitializedFieldError;

#[derive(PartialEq, Debug, Clone, Default)]
/// Builder for the camera, it exists for easy overriding camera options like width or height.
pub struct CameraBuilder {
    pub target_width: Option<usize>,
    pub target_height: Option<usize>,
    pub field_of_view: Option<f64>,
    view_transformation: Matrix,
}

impl CameraBuilder {
    pub fn target_width(&mut self, target_width: usize) -> &mut Self {
        self.target_width = Some(target_width);
        self
    }

    pub fn default_target_width(&mut self, target_width: usize) -> &mut Self {
        if self.target_width.is_none() {
            self.target_width = Some(target_width);
        }
        self
    }

    pub fn optional_target_width(&mut self, target_width: Option<usize>) -> &mut Self {
        if let Some(target_width) = target_width {
            self.target_width = Some(target_width);
        }
        self
    }

    pub fn target_height(&mut self, target_height: usize) -> &mut Self {
        self.target_height = Some(target_height);
        self
    }

    pub fn default_target_height(&mut self, target_height: usize) -> &mut Self {
        if self.target_height.is_none() {
            self.target_height = Some(target_height);
        }
        self
    }

    pub fn optional_target_height(&mut self, target_height: Option<usize>) -> &mut Self {
        if let Some(target_height) = target_height {
            self.target_height = Some(target_height);
        }
        self
    }

    pub fn field_of_view(&mut self, field_of_view: f64) -> &mut Self {
        self.field_of_view = Some(field_of_view);
        self
    }

    pub fn default_field_of_view(&mut self, field_of_view: f64) -> &mut Self {
        if self.field_of_view.is_none() {
            self.field_of_view = Some(field_of_view);
        }
        self
    }

    pub fn optional_field_of_view(&mut self, field_of_view: Option<f64>) -> &mut Self {
        if let Some(field_of_view) = field_of_view {
            self.field_of_view = Some(field_of_view);
        }
        self
    }

    pub fn view_transformation(&mut self, view_transformation: Matrix) -> &mut Self {
        self.view_transformation = view_transformation;
        self
    }
}

impl CameraBuilder {
    fn build_inverse_view_transformation(&self) -> Matrix {
        self.view_transformation
            .clone()
            .inverse()
            .expect("view transformation must be inversible")
    }

    pub fn build(&self) -> Result<Camera, CameraBuilderError> {
        let target_width =
            self.target_width
                .ok_or(derive_builder::UninitializedFieldError::from(
                    "target_width",
                ))?;

        let target_height =
            self.target_height
                .ok_or(derive_builder::UninitializedFieldError::from(
                    "target_height",
                ))?;

        let inverse_view = self.build_inverse_view_transformation();

        let field_of_view =
            self.field_of_view
                .ok_or(derive_builder::UninitializedFieldError::from(
                    "field_of_view",
                ))?;

        Ok(Camera::with_inverse_transformation(
            target_width,
            target_height,
            field_of_view,
            inverse_view,
        ))
    }
}

impl Camera {
    pub fn new(target_width: usize, target_height: usize, field_of_view: f64) -> Self {
        Self::with_transformation(
            target_width,
            target_height,
            field_of_view,
            Matrix::identity(),
        )
    }

    pub fn with_transformation(
        target_width: usize,
        target_height: usize,
        field_of_view: f64,
        transformation: Matrix,
    ) -> Self {
        let inverse_transformation = transformation
            .inverse()
            .expect("transformation matrix must be inversible");
        Self::with_inverse_transformation(
            target_width,
            target_height,
            field_of_view,
            inverse_transformation,
        )
    }

    pub fn with_inverse_transformation(
        target_width: usize,
        target_height: usize,
        field_of_view: f64,
        inverse_transformation: Matrix,
    ) -> Self {
        let half_view = (field_of_view / 2.).tan();
        assert!(target_height > 0);
        let h_v_aspect = target_width as f64 / target_height as f64;

        let (half_width, half_height) = match h_v_aspect >= 1. {
            true => (half_view, half_view / h_v_aspect),
            false => (half_view * h_v_aspect, half_view),
        };

        let pixel_size = 2. * half_width / target_width as f64;

        Self {
            target_width,
            target_height,
            inverse_transformation,

            pixel_size,
            half_width,
            half_height,
            field_of_view,
        }
    }

    pub fn ray_for_pixel(&self, x: f64, y: f64) -> Ray {
        let x_offset_to_center = (x + 0.5) * self.pixel_size;
        let y_offset_to_center = (y + 0.5) * self.pixel_size;

        let scene_x = self.half_width - x_offset_to_center;
        let scene_y = self.half_height - y_offset_to_center;

        let pixel = self.inverse_transformation * Point::new(scene_x, scene_y, -1.);
        let origin = self.inverse_transformation * Point::zero();
        let direction = pixel - origin;

        Ray::new(origin, direction.normalize())
    }

    pub fn image(&self) -> Image {
        Image::new(self.target_width, self.target_height)
    }

    pub fn target_width(&self) -> usize {
        self.target_width
    }

    pub fn target_height(&self) -> usize {
        self.target_height
    }

    pub fn field_of_view(&self) -> f64 {
        self.field_of_view
    }

    pub fn inverse_transformation(&self) -> Matrix {
        self.inverse_transformation
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4};

    use super::*;
    use crate::{
        assert_approx_eq_low_prec,
        math::{
            approx_eq::ApproxEq, color::Color, transform::Transform, tuple::Tuple, vector::Vector,
        },
        render::renderer::Renderer,
    };

    #[test]
    fn identity_matrix_is_default_transformation() {
        let camera = Camera::new(160, 120, FRAC_PI_2);

        assert_eq!(camera.inverse_transformation, Matrix::identity());
    }
    #[test]
    fn pixel_size_for_horizontal_image() {
        let camera = Camera::new(200, 125, FRAC_PI_2);

        assert!(camera.pixel_size.approx_eq(&0.01));
    }

    #[test]
    fn pixel_size_for_vertical_image() {
        let camera = Camera::new(125, 200, FRAC_PI_2);

        assert!(camera.pixel_size.approx_eq(&0.01));
    }

    #[test]
    fn construct_ray_thru_image_center() {
        let camera = Camera::new(201, 101, FRAC_PI_2);

        let ray = camera.ray_for_pixel(100., 50.);
        assert_eq!(ray.origin(), &Point::new(0., 0., 0.));
        assert_eq!(ray.direction(), &Vector::new(0., 0., -1.));
    }
    #[test]
    fn construct_ray_thru_image_corner() {
        let camera = Camera::new(201, 101, FRAC_PI_2);

        let ray = camera.ray_for_pixel(0., 0.);
        assert_approx_eq_low_prec!(ray.origin(), &Point::new(0., 0., 0.));
        assert_approx_eq_low_prec!(ray.direction(), &Vector::new(0.66519, 0.33259, -0.66851));
    }

    #[test]
    fn construct_ray_when_camera_is_transformed() {
        let camera = Camera::with_transformation(
            201,
            101,
            FRAC_PI_2,
            Matrix::translation(0., -2., 5.)
                .rotate_y(FRAC_PI_4)
                .transformed(),
        );

        let ray = camera.ray_for_pixel(100., 50.);
        assert_eq!(ray.origin(), &Point::new(0., 2., -5.));
        assert_eq!(
            ray.direction(),
            &Vector::new(FRAC_1_SQRT_2, 0., -FRAC_1_SQRT_2)
        );
    }

    #[test]
    fn render_scene_with_camera() {
        let from = Point::new(0., 0., -5.);
        let to = Point::new(0., 0., 0.);
        let up_v = Vector::new(0., 1., 0.);
        let camera = Camera::with_transformation(
            11,
            11,
            FRAC_PI_2,
            Matrix::view_tranformation(from, to, up_v),
        );
        let mut renderer = Renderer::default_testing(camera);

        let image = renderer.render();
        assert_approx_eq_low_prec!(image.pixel_at(5, 5), Color::new(0.38066, 0.47583, 0.2855));
    }
}
