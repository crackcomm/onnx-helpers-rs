//! ONNX node attribute helpers.

use onnx_pb::{GraphProto, TensorProto};

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
