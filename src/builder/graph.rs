//! Graph builder.

use onnx_pb::{GraphProto, NodeProto, TensorProto, ValueInfoProto};

/// Graph builder.
#[derive(Default, Clone)]
pub struct Graph {
    name: String,
    nodes: Vec<NodeProto>,
    inputs: Vec<ValueInfoProto>,
    outputs: Vec<ValueInfoProto>,
    initializers: Vec<TensorProto>,
    doc_string: Option<String>,
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

    /// Inserts graph nodes.
    #[inline]
    pub fn node<T: Into<NodeProto>>(mut self, node: T) -> Self {
        self.nodes.push(node.into());
        self
    }

    /// Inserts graph inputs.
    #[inline]
    pub fn input<T: Into<ValueInfoProto>>(mut self, input: T) -> Self {
        self.inputs.push(input.into());
        self
    }

    /// Inserts graph outputs.
    #[inline]
    pub fn output<T: Into<ValueInfoProto>>(mut self, output: T) -> Self {
        self.outputs.push(output.into());
        self
    }

    /// Inserts graph initializers.
    #[inline]
    pub fn initializer<T: Into<TensorProto>>(mut self, initializer: T) -> Self {
        self.initializers.push(initializer.into());
        self
    }

    /// Builds the graph.
    #[inline]
    pub fn build(self) -> GraphProto {
        GraphProto {
            name: self.name,
            node: self.nodes,
            input: self.inputs,
            output: self.outputs,
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
