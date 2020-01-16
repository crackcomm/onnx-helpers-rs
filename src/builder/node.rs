//! Node builder.

use onnx_pb::{AttributeProto, NodeProto};

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
    attributes: Vec<AttributeProto>,
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

    /// Inserts node inputs.
    #[inline]
    pub fn input<S: Into<String>>(mut self, input: S) -> Self {
        self.inputs.push(input.into());
        self
    }

    /// Inserts node outputs.
    #[inline]
    pub fn output<S: Into<String>>(mut self, output: S) -> Self {
        self.outputs.push(output.into());
        self
    }

    /// Inserts node attributes.
    #[inline]
    pub fn attribute<S: Into<String>, A: Into<Attribute>>(mut self, name: S, attribute: A) -> Self {
        self.attributes.push(make_attribute(name, attribute));
        self
    }

    /// Builds the node.
    #[inline]
    pub fn build(mut self) -> nodes::Node {
        let name = if let Some(name) = self.name {
            name
        } else {
            if self.inputs.len() == 2 {
                format!(
                    "{}_{}_{}",
                    self.inputs.get(0).unwrap(),
                    self.op_type,
                    self.inputs.get(1).unwrap()
                )
            } else {
                format!("{}_{}_N", self.op_type, self.inputs.join("_"))
            }
        };
        let output = if self.outputs.len() > 0 {
            self.outputs
        } else {
            vec![format!("OOF_{}", name)]
        };
        let proto = NodeProto {
            name,
            domain: self.domain.unwrap_or_default(),
            op_type: self.op_type,
            doc_string: self.doc_string.unwrap_or_default(),
            input: self.inputs,
            output: output,
            attribute: self.attributes,
        };
        let mut node = nodes::Node::from(proto);
        nodes::maybe_bag_node(self.bag.as_mut(), &mut node);
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