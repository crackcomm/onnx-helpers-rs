//! Graph builder.

use onnx_pb::{GraphProto, NodeProto, TensorProto, ValueInfoProto};

use crate::{
    builder::{self, Bag, Marker},
    nodes,
};

/// Graph builder.
#[derive(Default, Clone)]
pub struct Graph {
    name: String,
    nodes: Vec<NodeProto>,
    inputs: Vec<ValueInfoProto>,
    outputs: Vec<ValueInfoProto>,
    initializers: Vec<TensorProto>,
    doc_string: Option<String>,
    bag: Bag,
}

impl Graph {
    /// Creates a new builder.
    #[inline]
    pub fn new<S: Into<String>>(name: S) -> Self {
        Graph {
            name: name.into(),
            ..Graph::default()
        }
    }

    /// Sets graph name.
    #[inline]
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = name.into();
        self
    }

    /// Sets graph doc_string.
    #[inline]
    pub fn doc_string<S: Into<String>>(mut self, doc_string: S) -> Self {
        self.doc_string = Some(doc_string.into());
        self
    }

    /// Creates constant node in a graph.
    #[inline]
    pub fn constant<T: Into<nodes::Node>>(&mut self, node: T) -> nodes::Node {
        let node = node.into();
        self.nodes.push(node.clone().into());
        node
    }

    /// Inserts graph nodes.
    #[inline]
    pub fn nodes<T: Into<NodeProto>>(mut self, node: T) -> Self {
        self.nodes.push(node.into());
        self
    }

    /// Inserts graph inputs.
    #[inline]
    pub fn inputs<T: Into<ValueInfoProto>>(mut self, input: T) -> Self {
        self.inputs.push(input.into());
        self
    }

    /// Inserts graph outputs.
    #[inline]
    pub fn outputs<T: Into<ValueInfoProto>>(mut self, output: T) -> Self {
        self.outputs.push(output.into());
        self
    }

    /// Inserts graph initializers.
    #[inline]
    pub fn initializer<T: Into<TensorProto>>(mut self, initializer: T) -> Self {
        self.initializers.push(initializer.into());
        self
    }

    /// Creates graph node builder.
    #[inline]
    pub fn node<T: Into<String>>(&mut self, name: T) -> builder::Node {
        let mut node = builder::Node::default().name(name);
        node.bag = Some(self.bag.clone());
        node
    }

    /// Creates graph input builder.
    #[inline]
    pub fn input<T: Into<String>>(&mut self, name: T) -> builder::Value {
        let mut value = builder::Value::new(name);
        value.bag = Some(self.bag.clone());
        value.marker = Some(Marker::Input);
        value
    }

    /// Creates graph output builder.
    #[inline]
    pub fn output<T: Into<String>>(&mut self, name: T) -> builder::Value {
        let mut value = builder::Value::new(name);
        value.bag = Some(self.bag.clone());
        value.marker = Some(Marker::Output);
        value
    }

    /// Builds the graph.
    #[inline]
    pub fn build(self) -> GraphProto {
        let mut nodes = self.nodes;
        nodes.extend(self.bag.nodes().into_iter().map(Into::into));
        let mut inputs = self.inputs;
        inputs.extend(self.bag.inputs().into_iter().map(Into::into));
        let mut outputs = self.outputs;
        outputs.extend(self.bag.outputs().into_iter().map(Into::into));
        GraphProto {
            name: self.name,
            node: nodes,
            input: inputs,
            output: outputs,
            doc_string: self.doc_string.unwrap_or_default(),
            initializer: self.initializers,
            ..GraphProto::default()
        }
    }
}

impl Into<GraphProto> for Graph {
    fn into(self) -> GraphProto {
        self.build()
    }
}
