use std::collections::HashMap;

use crate::primitive::{point::Point, tuple::Tuple};

use super::object::{Object, ObjectGroup, Shape, Triangle};

pub struct ObjParser {
    ignored: usize,
    vertices: Vec<Point>,
    groups: HashMap<String, ObjectGroup>,
    curr_group: Option<String>,
    main_group: ObjectGroup,
}

impl ObjParser {
    pub fn new() -> ObjParser {
        ObjParser {
            ignored: 0,
            groups: HashMap::new(),
            vertices: Vec::new(),
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

    fn fan_triangulation(&mut self, vertex_indices: Vec<usize>) {
        let v = self.vertices[vertex_indices[0]];
        let triangles: Vec<Object> = vertex_indices
            .windows(2)
            .skip(1)
            .map(|slice| match slice {
                [id1, id2] => {
                    Object::with_shape(Shape::triangle(v, self.vertices[*id1], self.vertices[*id2]))
                }
                _ => unreachable!(),
            })
            .collect();
        self.curr_group_mut().add_children(triangles);
    }

    fn add_group(&mut self, name: String) {
        self.curr_group = Some(name);
    }

    fn add_vertex(&mut self, x: f64, y: f64, z: f64) {
        self.vertices.push(Point::new(x, y, z));
    }

    fn parse_line(&mut self, line: &str) -> Result<(), String> {
        if line.starts_with('#') || line.is_empty() {
            return Ok(());
        }

        let mut iter = line.split_whitespace();
        match iter.next() {
            Some("v") => {
                let x = iter.next().unwrap().parse().unwrap();
                let y = iter.next().unwrap().parse().unwrap();
                let z = iter.next().unwrap().parse().unwrap();
                self.add_vertex(x, y, z);
            }
            Some("f") => {
                let vertex_indices = iter
                    .map(|s| s.parse::<usize>().unwrap() - 1)
                    .collect::<Vec<usize>>();
                self.fan_triangulation(vertex_indices);
            }
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
            self.main_group.add_child(group.into_object());
        });
        self.main_group.clone()
    }

    pub fn parse_to_object(source: String) -> Result<Object, String> {
        Self::default()
            .parse(source)
            .map(|group| group.into_object())
    }
}

impl Default for ObjParser {
    fn default() -> Self {
        Self::new()
    }
}

mod tests {
    use crate::render::object::{Object, Shape};

    use super::*;

    #[test]
    fn ignoring_unrecognized_lines() {
        let data = r#"
            There was a young lady named Bright
            who traveled much faster than light.
            She set out one day
            in a relative way,
            and came back the previous night.
        "#;
        let mut parser = ObjParser::new();
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
        let mut parser = ObjParser::new();
        parser.not_consuming_parse(data.to_string()).unwrap();

        assert_eq!(parser.vertices().len(), 4);
        assert_eq!(parser.vertices()[0], Point::new(-1.0, 1.0, 0.0));
        assert_eq!(parser.vertices()[1], Point::new(-1.0, 0.5, 0.0));
        assert_eq!(parser.vertices()[2], Point::new(1.0, 0.0, 0.0));
        assert_eq!(parser.vertices()[3], Point::new(1.0, 1.0, 0.0));
    }

    fn _obj_as_triangle(object: &Object) -> Option<Triangle> {
        match object.shape() {
            Shape::Triangle(t) => Some(t.clone()),
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
        let mut parser = ObjParser::new();
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
        let mut parser = ObjParser::new();
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
        let mut parser = ObjParser::new();

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
        assert!(matches!(children[0].shape(), &Shape::Group(_)));
        assert!(matches!(children[1].shape(), &Shape::Group(_)));
    }
}
