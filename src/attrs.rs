//! ONNX node attribute helpers.

use onnx_pb::{attribute_proto::AttributeType, AttributeProto, GraphProto, TensorProto};

/// Attribute constructor.
pub enum Attribute {
    Float(f32),
    Floats(Vec<f32>),
    Int(i64),
    Ints(Vec<i64>),
    Bytes(Vec<u8>),
    String(String),
    Strings(Vec<String>),
    Tensor(TensorProto),
    Tensors(Vec<TensorProto>),
    Graph(GraphProto),
    Graphs(Vec<GraphProto>),
}

macro_rules! attr_converter {
    ( $a:ident, $b:ty ) => {
        impl From<$b> for Attribute {
            fn from(v: $b) -> Self {
                Attribute::$a(v)
            }
        }
    };
}

impl From<bool> for Attribute {
    fn from(v: bool) -> Self {
        Attribute::Int(if v { 1 } else { 0 })
    }
}

attr_converter!(Float, f32);
attr_converter!(Floats, Vec<f32>);
attr_converter!(Int, i64);
attr_converter!(Bytes, Vec<u8>);
attr_converter!(String, String);
attr_converter!(Strings, Vec<String>);
attr_converter!(Ints, Vec<i64>);
attr_converter!(Tensor, TensorProto);
attr_converter!(Tensors, Vec<TensorProto>);
attr_converter!(Graph, GraphProto);
attr_converter!(Graphs, Vec<GraphProto>);

impl From<&str> for Attribute {
    fn from(v: &str) -> Self {
        v.to_owned().into()
    }
}

impl From<Vec<&str>> for Attribute {
    fn from(v: Vec<&str>) -> Self {
        v.into_iter()
            .map(|s| s.to_owned())
            .collect::<Vec<_>>()
            .into()
    }
}

/// Creates a new attribute struct.
pub(crate) fn make_attribute<S: Into<String>, A: Into<Attribute>>(
    name: S,
    attribute: A,
) -> AttributeProto {
    let mut attr_proto = AttributeProto {
        name: name.into(),
        ..AttributeProto::default()
    };
    match attribute.into() {
        Attribute::Float(val) => {
            attr_proto.f = val;
            attr_proto.r#type = AttributeType::Float as i32;
        }
        Attribute::Floats(vals) => {
            attr_proto.floats = vals;
            attr_proto.r#type = AttributeType::Floats as i32;
        }
        Attribute::Int(val) => {
            attr_proto.i = val;
            attr_proto.r#type = AttributeType::Int as i32;
        }
        Attribute::Ints(vals) => {
            attr_proto.ints = vals;
            attr_proto.r#type = AttributeType::Ints as i32;
        }
        Attribute::Bytes(val) => {
            attr_proto.s = val;
            attr_proto.r#type = AttributeType::String as i32;
        }
        Attribute::String(val) => {
            attr_proto.s = val.into();
            attr_proto.r#type = AttributeType::String as i32;
        }
        Attribute::Strings(vals) => {
            attr_proto.strings = vals.into_iter().map(Into::into).collect();
            attr_proto.r#type = AttributeType::Strings as i32;
        }
        Attribute::Graph(val) => {
            attr_proto.g = Some(val);
            attr_proto.r#type = AttributeType::Graph as i32;
        }
        Attribute::Graphs(vals) => {
            attr_proto.graphs = vals;
            attr_proto.r#type = AttributeType::Graphs as i32;
        }
        Attribute::Tensor(val) => {
            attr_proto.t = Some(val);
            attr_proto.r#type = AttributeType::Tensor as i32;
        }
        Attribute::Tensors(vals) => {
            attr_proto.tensors = vals;
            attr_proto.r#type = AttributeType::Tensors as i32;
        }
    };
    attr_proto
}
