use std::collections::HashMap;

use crate::{
    math::{point::Point, tuple::Tuple, vector::Vector},
    scene::object::{group::ObjectGroup, primitive::shape::Shape, Object},
};

pub struct ObjModelParser {
    ignored: usize,
    vertices: Vec<Point>,
    normals: Vec<Vector>,
    groups: HashMap<String, ObjectGroup>,
    curr_group: Option<String>,
    main_group: ObjectGroup,
}

impl ObjModelParser {
    pub fn new() -> ObjModelParser {
        ObjModelParser {
            ignored: 0,
            groups: HashMap::new(),
            vertices: Vec::new(),
            normals: Vec::new(),
            curr_group: None,
            main_group: ObjectGroup::default(),
        }
    }

    pub fn ignored(&self) -> usize {
        self.ignored
    }

    pub fn vertices(&self) -> &[Point] {
        self.vertices.as_ref()
    }

    fn curr_group_mut(&mut self) -> &mut ObjectGroup {
        match self.curr_group {
            Some(ref curr_group) => self.groups.entry(curr_group.clone()).or_default(),
            None => &mut self.main_group,
        }
    }

    fn fan_triangulation(
        &mut self,
        vertex_indices: Vec<usize>,
        normal_indices: Option<Vec<usize>>,
    ) {
        let v = self.vertices[vertex_indices[0]];
        let triangle_iter: Vec<Object> = match normal_indices {
            None => vertex_indices
                .windows(2)
                .skip(1)
                .map(|slice| match slice {
                    [id1, id2] => Object::primitive_with_shape(Shape::triangle(
                        v,
                        self.vertices[*id1],
                        self.vertices[*id2],
                    )),
                    _ => unreachable!(),
                })
                .collect(),
            Some(normal_indices) => {
                let n = self.normals[normal_indices[0]];
                assert_eq!(vertex_indices.len(), normal_indices.len());

                vertex_indices
                    .iter()
                    .zip(normal_indices.iter())
                    .enumerate()
                    .skip(2)
                    .map(|(index_id, pair)| {
                        let v1 = vertex_indices[index_id - 1];
                        let n1 = normal_indices[index_id - 1];

                        let v2 = *pair.0;
                        let n2 = *pair.1;

                        Object::primitive_with_shape(Shape::smooth_triangle(
                            v,
                            self.vertices[v1],
                            self.vertices[v2],
                            n,
                            self.normals[n1],
                            self.normals[n2],
                        ))
                    })
                    .collect()
            }
        };
        self.curr_group_mut().add_children(triangle_iter);
    }

    fn face<'a>(&'a mut self, iter: &mut impl Iterator<Item = &'a str>) {
        let mut vertex_indices = Vec::new();
        let mut normal_indices = Vec::new();

        for face in iter {
            let mut indices = face.split('/');
            let vertex_index = indices.next().unwrap().parse::<usize>().unwrap() - 1;
            // skip texture index
            indices.next();
            vertex_indices.push(vertex_index);
            if let Some(normal_index) = indices.next() {
                normal_indices.push(normal_index.parse::<usize>().unwrap() - 1);
            }
        }
        self.fan_triangulation(
            vertex_indices,
            if normal_indices.is_empty() {
                None
            } else {
                Some(normal_indices)
            },
        );
    }

    fn add_group(&mut self, name: String) {
        self.curr_group = Some(name);
    }

    fn add_vertex(&mut self, tuple: (f64, f64, f64)) {
        let (x, y, z) = tuple;
        self.vertices.push(Point::new(x, y, z));
    }

    fn add_normal(&mut self, tuple: (f64, f64, f64)) {
        let (x, y, z) = tuple;
        self.normals.push(Vector::new(x, y, z));
    }

    fn parse_tuple<'a>(&'a self, iter: &mut impl Iterator<Item = &'a str>) -> (f64, f64, f64) {
        let x = iter.next().unwrap().parse().unwrap();
        let y = iter.next().unwrap().parse().unwrap();
        let z = iter.next().unwrap().parse().unwrap();
        (x, y, z)
    }

    fn parse_line(&mut self, line: &str) -> Result<(), String> {
        if line.starts_with('#') || line.is_empty() {
            return Ok(());
        }

        let mut iter = line.split_whitespace();
        match iter.next() {
            Some("v") => {
                self.add_vertex(self.parse_tuple(&mut iter));
            }
            Some("vn") => {
                self.add_normal(self.parse_tuple(&mut iter));
            }

            Some("f") => self.face(&mut iter),
            Some("g") => {
                self.add_group(iter.next().unwrap().to_string());
            }
            _ => {
                self.ignored += 1;
            }
        }
        Ok(())
    }

    pub fn not_consuming_parse(&mut self, source: String) -> Result<(), String> {
        source
            .lines()
            .try_for_each(|line| self.parse_line(line.trim()))?;
        Ok(())
    }

    pub fn parse(mut self, source: String) -> Result<ObjectGroup, String> {
        self.not_consuming_parse(source)?;
        Ok(self.into_group())
    }

    pub fn into_group(mut self) -> ObjectGroup {
        self.groups.into_iter().for_each(|(_, group)| {
            self.main_group.add_child(group.into());
        });
        self.main_group.clone()
    }

    pub fn parse_to_object(source: String) -> Result<Object, String> {
        Self::default().parse(source).map(|group| group.into())
    }
}

impl Default for ObjModelParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        approx_eq::ApproxEq,
        assert_approx_eq_low_prec,
        scene::object::primitive::{smooth_triangle::SmoothTriangle, triangle::Triangle},
    };

    #[test]
    fn ignoring_unrecognized_lines() {
        let data = r#"
            There was a young lady named Bright
            who traveled much faster than light.
            She set out one day
            in a relative way,
            and came back the previous night.
        "#;
        let mut parser = ObjModelParser::new();
        parser.not_consuming_parse(data.to_string()).unwrap();
        assert_eq!(parser.ignored(), 5);
    }

    #[test]
    fn parser_records_vertices() {
        let data = r#"
            v -1 1 0
            v -1.0000 0.5000 0.0000
            v 1 0 0
            v 1 1 0
        "#;
        let mut parser = ObjModelParser::new();
        parser.not_consuming_parse(data.to_string()).unwrap();

        assert_eq!(parser.vertices().len(), 4);
        assert_approx_eq_low_prec!(parser.vertices()[0], Point::new(-1.0, 1.0, 0.0));
        assert_approx_eq_low_prec!(parser.vertices()[1], Point::new(-1.0, 0.5, 0.0));
        assert_approx_eq_low_prec!(parser.vertices()[2], Point::new(1.0, 0.0, 0.0));
        assert_approx_eq_low_prec!(parser.vertices()[3], Point::new(1.0, 1.0, 0.0));
    }

    fn _obj_as_triangle(object: &Object) -> Option<Triangle> {
        match object.as_primitive().map(|p| p.shape()) {
            Some(Shape::Triangle(t)) => Some(t.clone()),
            _ => None,
        }
    }

    fn _group_as_triangles(group: ObjectGroup) -> Vec<Triangle> {
        group
            .children()
            .iter()
            .map(|child| _obj_as_triangle(child).unwrap())
            .collect()
    }

    #[test]
    fn parsing_triangle_faces() {
        let data = r#"
            v -1 1 0
            v -1 0 0
            v 1 0 0
            v 1 1 0

            f 1 2 3
            f 1 3 4
        "#;
        let mut parser = ObjModelParser::new();
        parser.not_consuming_parse(data.to_string()).unwrap();
        let vertices = parser.vertices();

        let t1 = Triangle::new(vertices[0], vertices[1], vertices[2]);
        let t2 = Triangle::new(vertices[0], vertices[2], vertices[3]);
        let group = parser.into_group();
        let children = group.children();

        assert_eq!(children.len(), 2);
        assert_eq!(_obj_as_triangle(&children[0]).unwrap(), t1);
        assert_eq!(_obj_as_triangle(&children[1]).unwrap(), t2);
    }

    #[test]
    fn triangulating_polygons() {
        let data = r#"
            v -1 1 0
            v -1 0 0
            v 1 0 0
            v 1 1 0
            v 0 2 0
            f 1 2 3 4 5
        "#;
        let mut parser = ObjModelParser::new();
        parser.not_consuming_parse(data.to_string()).unwrap();
        let vertices = parser.vertices();

        let t1 = Triangle::new(vertices[0], vertices[1], vertices[2]);
        let t2 = Triangle::new(vertices[0], vertices[2], vertices[3]);
        let t3 = Triangle::new(vertices[0], vertices[3], vertices[4]);
        let group = parser.into_group();
        let children = group.children();

        assert_eq!(children.len(), 3);
        assert_eq!(_obj_as_triangle(&children[0]).unwrap(), t1);
        assert_eq!(_obj_as_triangle(&children[1]).unwrap(), t2);
        assert_eq!(_obj_as_triangle(&children[2]).unwrap(), t3);
    }

    #[test]
    fn triangles_in_groups() {
        let mut parser = ObjModelParser::new();

        let data: &str = r#"
        v -1 1 0
        v -1 0 0
        v 1 0 0
        v 1 1 0

        g FirstGroup
        f 1 2 3
        g SecondGroup
        f 1 3 4
        "#;

        parser.not_consuming_parse(data.to_string()).unwrap();
        let vertices = parser.vertices();

        let g1 = parser.groups.get("FirstGroup").unwrap();
        let g2 = parser.groups.get("SecondGroup").unwrap();

        let t1 = Triangle::new(vertices[0], vertices[1], vertices[2]);
        let t2 = Triangle::new(vertices[0], vertices[2], vertices[3]);

        let g1_exp = vec![t1];
        let g2_exp = vec![t2];

        assert_eq!(_group_as_triangles(g1.clone()), g1_exp);
        assert_eq!(_group_as_triangles(g2.clone()), g2_exp);

        let group = parser.into_group();
        let children = group.children();

        assert_eq!(children.len(), 2);
        assert!(children[0].as_group().is_some());
        assert!(children[1].as_group().is_some());
    }

    #[test]
    fn vertex_normal_records() {
        let data = r#"
            vn 0 0 1
            vn 0.707 0 -0.707
            vn 1 2 3
        "#;
        let mut parser = ObjModelParser::new();
        parser.not_consuming_parse(data.to_string()).unwrap();

        assert_eq!(parser.ignored(), 0);
        assert_eq!(parser.normals.len(), 3);
        assert_approx_eq_low_prec!(parser.normals[0], Vector::new(0.0, 0.0, 1.0));
        assert_approx_eq_low_prec!(parser.normals[1], Vector::new(0.707, 0.0, -0.707));
        assert_approx_eq_low_prec!(parser.normals[2], Vector::new(1.0, 2.0, 3.0));
    }

    fn _obj_as_smooth_triangle(object: &Object) -> SmoothTriangle {
        match object.as_primitive().map(|p| p.shape()) {
            Some(Shape::SmoothTriangle(t)) => t.clone(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn faces_with_normals() {
        let data = r#"
            v 0 1 0
            v -1 0 0
            v 1 0 0
            vn -1 0 0
            vn 1 0 0
            vn 0 1 0
            f 1//3 2//1 3//2
            f 1/0/3 2/0/1 3/0/2
        "#;
        let mut parser = ObjModelParser::new();
        parser.not_consuming_parse(data.to_string()).unwrap();

        let child1 = _obj_as_smooth_triangle(&parser.main_group.children()[0]);
        let child2 = _obj_as_smooth_triangle(&parser.main_group.children()[1]);

        assert_eq!(child1, child2);
        assert_approx_eq_low_prec!(child1.p1(), parser.vertices[0]);
        assert_approx_eq_low_prec!(child1.p2(), parser.vertices[1]);
        assert_approx_eq_low_prec!(child1.p3(), parser.vertices[2]);
        assert_approx_eq_low_prec!(child1.n1(), parser.normals[2]);
        assert_approx_eq_low_prec!(child1.n2(), parser.normals[0]);
        assert_approx_eq_low_prec!(child1.n3(), parser.normals[1]);
    }
}
