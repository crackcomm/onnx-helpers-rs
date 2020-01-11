//! ONNX model construction helpers.

mod attrs;

pub use self::attrs::*;

use onnx_pb::attribute_proto::AttributeType;
use onnx_pb::tensor_proto::DataType;
use onnx_pb::tensor_shape_proto::Dimension;
use onnx_pb::type_proto::{self, Tensor};
use onnx_pb::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, StringStringEntryProto,
    TensorProto, TensorShapeProto, TypeProto, ValueInfoProto, Version,
};

const DEFAULT_OPSET_ID_VERSION: i64 = 11;

/// Creates a new model struct.
pub fn make_model<S: Into<String>>(
    graph: GraphProto,
    domain: Option<S>,
    model_version: Option<i64>,
    producer_name: Option<S>,
    producer_version: Option<S>,
    doc_string: Option<S>,
    metadata: Option<Vec<(S, S)>>,
    opset_imports: Option<Vec<OperatorSetIdProto>>,
) -> ModelProto {
    let opset_import = opset_imports.unwrap_or_else(|| {
        vec![OperatorSetIdProto {
            domain: String::default(),
            version: DEFAULT_OPSET_ID_VERSION,
        }]
    });
    let metadata_props = metadata
        .map(|metadata| {
            metadata
                .into_iter()
                .map(|(k, v)| StringStringEntryProto {
                    key: k.into(),
                    value: v.into(),
                })
                .collect()
        })
        .unwrap_or_default();
    ModelProto {
        ir_version: Version::IrVersion as i64,
        graph: Some(graph),
        domain: unwrap_or_default(domain),
        doc_string: unwrap_or_default(doc_string),
        producer_name: unwrap_or_default(producer_name),
        producer_version: unwrap_or_default(producer_version),
        model_version: unwrap_or_default(model_version),
        opset_import,
        metadata_props,
        ..ModelProto::default()
    }
}

/// Creates a new graph struct.
pub fn make_graph<S: Into<String>>(
    nodes: Vec<NodeProto>,
    name: S,
    inputs: Vec<ValueInfoProto>,
    outputs: Vec<ValueInfoProto>,
    initializer: Vec<TensorProto>,
    doc_string: Option<S>,
) -> GraphProto {
    GraphProto {
        name: name.into(),
        node: nodes,
        input: inputs,
        output: outputs,
        doc_string: unwrap_or_default(doc_string),
        initializer,
        ..GraphProto::default()
    }
}

/// Creates a new node struct.
pub fn make_node<S: Into<String>>(
    op_type: S,
    inputs: Vec<S>,
    outputs: Vec<S>,
    name: Option<S>,
    doc_string: Option<S>,
    domain: Option<S>,
    attributes: Vec<AttributeProto>,
) -> NodeProto {
    NodeProto {
        name: unwrap_or_default(name),
        domain: unwrap_or_default(domain),
        op_type: op_type.into(),
        doc_string: unwrap_or_default(doc_string),
        input: inputs.into_iter().map(|dim| dim.into()).collect(),
        output: outputs.into_iter().map(|dim| dim.into()).collect(),
        attribute: attributes,
    }
}

/// Creates a new tensor value information struct.
pub fn make_tensor_value_info<S: Into<String>, D: Into<Dimension>>(
    name: S,
    elem_type: DataType,
    shape: Vec<D>,
    doc_string: Option<S>,
) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            denotation: String::default(),
            value: Some(type_proto::Value::TensorType(Tensor {
                shape: Some(TensorShapeProto {
                    dim: shape.into_iter().map(|dim| dim.into()).collect(),
                }),
                elem_type: elem_type as i32,
            })),
        }),
        doc_string: unwrap_or_default(doc_string),
    }
}

/// Creates a new attribute struct.
pub fn make_attribute<S: Into<String>>(name: S, attribute: Attribute) -> AttributeProto {
    let mut attr_proto = AttributeProto {
        name: name.into(),
        ..AttributeProto::default()
    };
    match attribute {
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

#[inline(always)]
fn unwrap_or_default<V: Default, S: Into<V>>(s: Option<S>) -> V {
    s.map(|s| s.into()).unwrap_or_else(|| V::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    use prost::Message;

    #[test]
    fn compare_with_py_output() {
        let from_python = ModelProto::decode(&read_buf("tests/model.onnx")).unwrap();
        let x_input = make_tensor_value_info("X", DataType::Float, vec![1, 10], None);
        let mean_reduce = make_node(
            "ReduceMean",
            vec!["X"],
            vec!["Z"],
            None,
            None,
            None,
            vec![make_attribute("axes", vec![1i64].into())],
        );

        let graph = make_graph(
            vec![mean_reduce],
            "reduce-mean",
            vec![x_input],
            vec![],
            vec![],
            None,
        );

        let model = make_model(
            graph,
            None,            // domain
            None,            // model_version
            Some("reducer"), // producer_name
            None,            // producer_version
            None,            // doc_string
            None,            // metadata
            None,            // opset_imports
        );
        assert_eq!(model, from_python);
    }

    fn read_buf<P: AsRef<std::path::Path>>(path: P) -> Vec<u8> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).unwrap();
        let mut buffer = Vec::new();
        // read the whole file
        file.read_to_end(&mut buffer).unwrap();
        buffer
    }
}
