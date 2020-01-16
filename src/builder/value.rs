//! Value info builder.

use onnx_pb::{
    tensor_proto::DataType,
    tensor_shape_proto::Dimension,
    type_proto::{self, Tensor},
    TensorShapeProto, TypeProto, ValueInfoProto,
};

use crate::{
    builder::{Bag, Marker, Node},
    nodes,
};

/// Value info builder.
#[derive(Default, Clone)]
pub struct Value {
    name: String,
    elem_type: DataType,
    shape: Vec<Dimension>,
    doc_string: Option<String>,
    pub(crate) bag: Option<Bag>,
    pub(crate) marker: Option<Marker>,
}

impl Value {
    /// Creates a new builder.
    #[inline]
    pub fn new<S: Into<String>>(name: S) -> Self {
        Value {
            name: name.into(),
            ..Value::default()
        }
    }

    /// Sets value name.
    #[inline]
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = name.into();
        self
    }

    /// Sets value element type.
    #[inline]
    pub fn typed<T: Into<DataType>>(mut self, elem_type: T) -> Self {
        self.elem_type = elem_type.into();
        self
    }

    /// Sets value shape.
    #[inline]
    pub fn shape<D: Into<Dimension>>(mut self, shape: Vec<D>) -> Self {
        self.shape = shape.into_iter().map(|dim| dim.into()).collect();
        self
    }

    /// Inserts value dimension.
    #[inline]
    pub fn dim<D: Into<Dimension>>(mut self, dim: D) -> Self {
        self.shape.push(dim.into());
        self
    }

    /// Creates node for input.
    /// Requires builder to be bagged.
    #[inline]
    pub fn node(self) -> nodes::Node {
        let mut node = Node::named(self.name.clone()).build();
        node.bag = self.bag.clone();
        let marker = self.marker.as_ref().unwrap().clone();
        let mut bag: Bag = self.bag.as_ref().unwrap().clone();
        let value = self.build();
        bag.value(value, marker);
        node
    }

    /// Builds the value info.
    #[inline]
    pub fn build(self) -> ValueInfoProto {
        ValueInfoProto {
            name: self.name,
            r#type: Some(TypeProto {
                denotation: String::default(),
                value: Some(type_proto::Value::TensorType(Tensor {
                    shape: Some(TensorShapeProto { dim: self.shape }),
                    elem_type: self.elem_type as i32,
                })),
            }),
            doc_string: self.doc_string.unwrap_or_default(),
        }
    }
}

impl Into<ValueInfoProto> for Value {
    fn into(self) -> ValueInfoProto {
        self.build()
    }
}
