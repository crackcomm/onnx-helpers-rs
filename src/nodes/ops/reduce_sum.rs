//! Reduce sum operation.

use crate::{attrs::Attribute, builder, node_to_inner, nodes::Node};

/// Reduce sum node.
pub struct ReduceSum {
    inner: Node,
}

impl ReduceSum {
    /// Creates new reduce sum operation.
    pub fn new<S: Into<String>, A: Into<Attribute>>(input: S, axes: A, keepdims: bool) -> Self {
        ReduceSum {
            inner: builder::Node::new("ReduceSum")
                .input(input)
                .attribute("axes", axes)
                .attribute("keepdims", keepdims)
                .build(),
        }
    }
}

node_to_inner!(ReduceSum);
