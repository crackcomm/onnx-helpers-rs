//! Reduce mean operation.

use crate::{attrs::Attribute, builder, node_to_inner, nodes::Node};

/// Reduce mean node.
pub struct ReduceMean {
    inner: Node,
}

impl ReduceMean {
    /// Creates new reduce mean operation.
    pub fn new<S: Into<String>, A: Into<Attribute>>(input: S, axes: A, keepdims: bool) -> Self {
        ReduceMean {
            inner: builder::Node::new("ReduceMean")
                .input(input)
                .attribute("axes", axes)
                .attribute("keepdims", keepdims)
                .build(),
        }
    }
}

node_to_inner!(ReduceMean);
