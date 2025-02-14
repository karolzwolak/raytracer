use crate::render::{camera::Camera, world::World};

pub enum YamlParseError {}

pub fn parse_str(source: &str, width: usize, height: usize, fov: f64) -> (World, Camera) {
    let world = World::empty();
    let camera = Camera::new(width, height, fov);
    (world, camera)
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
intensity: [ 1, 1, 1 ]"#;

    #[test]
    fn empty_yaml() {
        let (world, camera) = parse_str("", WIDTH, HEIGHT, FOV);
        assert_eq!(world, World::empty());
        assert_eq!(camera, Camera::new(WIDTH, HEIGHT, FOV));
    }
    #[test]
    fn comments_are_supported() {
        let (world, camera) = parse_str(COMMENT_YAML, WIDTH, HEIGHT, FOV);
        assert_eq!(world, World::empty());
        assert_eq!(camera, Camera::new(WIDTH, HEIGHT, FOV));
    }
}
