use super::shape::Shape;

pub struct Object {
    shape: Shape,
}

impl Object {
    pub fn new(shape: Shape) -> Self {
        Self { shape }
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}
