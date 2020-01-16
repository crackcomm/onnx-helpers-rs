//! Constant operation.

use crate::{attrs::Attribute, builder, node_to_inner, nodes::Node};

/// Constant node.
pub struct Constant {
    inner: Node,
}

impl Constant {
    /// Creates new Constant operation.
    pub fn new<N: Into<String>, T: Into<Attribute>>(name: N, value: T) -> Self {
        Constant {
            inner: builder::Node::new("Constant")
                .name(name)
                .attribute("value", value)
                .build(),
        }
    }
}

node_to_inner!(Constant);
