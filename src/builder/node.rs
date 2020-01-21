//! Node builder.

use onnx_pb::NodeProto;

use crate::{
    attrs::{make_attribute, Attribute},
    builder::Bag,
    nodes,
};

/// Node builder.
#[derive(Default, Clone)]
pub struct Node {
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    name: Option<String>,
    doc_string: Option<String>,
    domain: Option<String>,
    attributes: Vec<(String, Attribute)>,
    pub(crate) bag: Option<Bag>,
}

impl Node {
    /// Creates a new builder.
    #[inline]
    pub fn new<S: Into<String>>(op_type: S) -> Self {
        Node {
            op_type: op_type.into(),
            ..Node::default()
        }
    }

    /// Creates a new builder.
    #[inline]
    pub fn named<S: Into<String>>(name: S) -> Self {
        Node {
            name: Some(name.into()),
            ..Node::default()
        }
    }

    /// Sets node name.
    #[inline]
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets node op type.
    #[inline]
    pub fn op<S: Into<String>>(mut self, op: S) -> Self {
        self.op_type = op.into();
        self
    }

    /// Sets node doc_string.
    #[inline]
    pub fn doc_string<S: Into<String>>(mut self, doc_string: S) -> Self {
        self.doc_string = Some(doc_string.into());
        self
    }

    /// Sets node domain.
    #[inline]
    pub fn domain<S: Into<String>>(mut self, domain: S) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Inserts node input.
    #[inline]
    pub fn input<S: Into<String>>(mut self, input: S) -> Self {
        self.inputs.push(input.into());
        self
    }

    /// Inserts node output.
    #[inline]
    pub fn output<S: Into<String>>(mut self, output: S) -> Self {
        self.outputs.push(output.into());
        self
    }

    /// Inserts node inputs.
    #[inline]
    pub fn inputs<I>(mut self, inputs: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        for input in inputs {
            self.inputs.push(input.into());
        }
        self
    }

    /// Inserts node outputs.
    #[inline]
    pub fn outputs<I>(mut self, outputs: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        for output in outputs {
            self.outputs.push(output.into());
        }
        self
    }

    /// Inserts node attributes.
    #[inline]
    pub fn attribute<S: Into<String>, A: Into<Attribute>>(mut self, name: S, attribute: A) -> Self {
        self.attributes.push((name.into(), attribute.into()));
        self
    }

    /// Builds the node.
    #[inline]
    pub fn build(self) -> nodes::Node {
        let name = if let Some(name) = self.name {
            name
        } else {
            let attrs = self
                .attributes
                .iter()
                .map(|(name, attr)| format!("{}_{}", name, attr))
                .collect::<Vec<String>>()
                .join("_");
            if self.inputs.len() == 2 {
                format!(
                    "{}_{}_{}_{}",
                    self.inputs.get(0).unwrap(),
                    self.op_type,
                    self.inputs.get(1).unwrap(),
                    attrs
                )
            } else {
                format!(
                    "S{}_{}_{}_{}E",
                    self.op_type,
                    self.inputs.join("_"),
                    self.op_type,
                    attrs
                )
            }
        };
        let output = if self.outputs.len() > 0 {
            self.outputs
        } else {
            vec![format!("{}O", name)]
        };
        let attributes = self
            .attributes
            .into_iter()
            .map(|(name, attr)| make_attribute(name, attr))
            .collect();
        let proto = NodeProto {
            name,
            domain: self.domain.unwrap_or_default(),
            op_type: self.op_type,
            doc_string: self.doc_string.unwrap_or_default(),
            input: self.inputs,
            output: output,
            attribute: attributes,
        };
        let mut node = nodes::Node::from_proto(proto);
        nodes::maybe_bag_node(self.bag.clone(), &mut node);
        node
    }
}

impl Into<nodes::Node> for Node {
    fn into(self) -> nodes::Node {
        self.build()
    }
}

impl Into<NodeProto> for Node {
    fn into(self) -> NodeProto {
        self.build().into()
    }
}
