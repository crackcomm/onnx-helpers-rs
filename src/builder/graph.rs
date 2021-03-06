//! Graph builder.

use onnx_pb::{GraphProto, NodeProto, TensorProto, TypeProto, ValueInfoProto};

use crate::{
    builder::{self, Bag, Marker},
    nodes::*,
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
    constants: i64,
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
    pub fn constant<S: Into<String>, T: Into<TensorProto>>(&mut self, name: S, tensor: T) -> Node {
        let mut node: Node = ops::Constant::new(name, tensor).into();
        node.bag = Some(self.bag.clone());
        self.bag.node(node.inner.clone());
        self.constants += 1;
        node
    }

    /// Creates a concat node in a graph.
    #[inline(always)]
    pub fn concat<I>(&mut self, axis: i64, inputs: I) -> Node
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        let mut node: Node = ops::Concat::new(axis, inputs).into();
        node.bag = Some(self.bag.clone());
        self.bag.node(node.inner.clone());
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

    /// Inserts typed graph outputs.
    #[inline]
    pub fn outputs_typed<T: Into<ValueInfoProto>, D: Into<TypeProto>>(
        mut self,
        output: T,
        typ: D,
    ) -> Self {
        let mut info = output.into();
        if info.r#type.is_none() {
            info.r#type = Some(typ.into());
        }
        self.outputs.push(info);
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

    /// Builds a model builder from graph.
    #[inline]
    pub fn model(self) -> builder::Model {
        builder::Model::new(self.build())
    }

    /// Builds the graph.
    #[inline]
    pub fn build(self) -> GraphProto {
        let mut nodes = self.nodes;
        nodes.extend(self.bag.nodes().into_iter().map(Into::into));
        nodes.dedup_by(|a, b| a.name == b.name);
        sort_nodes(&mut nodes);
        let mut inputs = self.inputs;
        inputs.extend(self.bag.inputs().into_iter().map(Into::into));
        inputs.dedup_by(|a, b| a.name == b.name);
        let mut outputs = self.outputs;
        outputs.extend(self.bag.outputs().into_iter().map(Into::into));
        outputs.dedup_by(|a, b| a.name == b.name);
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

fn sort_nodes(nodes: &mut Vec<NodeProto>) {
    use std::collections::HashMap;
    use std::iter::FromIterator;

    use petgraph::{algo::toposort, graphmap::DiGraphMap};

    let mut g = DiGraphMap::new();
    for node in nodes.iter() {
        for input in node.input.iter() {
            g.add_edge(input, &node.name, 1);
        }
        for output in node.output.iter() {
            g.add_edge(&node.name, output, 1);
        }
    }
    let sorted: HashMap<String, usize> = HashMap::from_iter(
        toposort(&g, None)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s.to_owned(), i)),
    );
    nodes.sort_by(|a, b| {
        let a = sorted.get(&a.name).unwrap();
        let b = sorted.get(&b.name).unwrap();
        a.partial_cmp(b).unwrap()
    });
}
