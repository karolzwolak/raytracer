pub mod approx_eq;

pub mod primitive {
    pub mod matrix2;
    pub mod matrix3;
    pub mod matrix4;
    pub mod point;
    pub mod tuple;
    pub mod vector;
}

pub mod render {
    pub mod canvas;
    pub mod color;
    pub mod intersection;
    pub mod object;
    pub mod ray;
    pub mod shape;
}

pub mod transformation;

pub mod playing_around {
    pub mod projectiles;
    pub mod rays_and_spheres;
    pub mod transformations;
}
