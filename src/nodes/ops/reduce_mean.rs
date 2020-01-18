//! Reduce mean operation.

use crate::{
    builder, node_to_inner,
    nodes::{Axes, Node},
};

/// Reduce mean node.
pub struct ReduceMean {
    inner: Node,
}

impl ReduceMean {
    /// Creates new reduce mean operation.
    pub fn new<S: Into<String>, A: Into<Axes>>(input: S, axes: A, keepdims: bool) -> Self {
        ReduceMean {
            inner: builder::Node::new("ReduceMean")
                .input(input)
                .attribute("axes", axes.into())
                .attribute("keepdims", keepdims)
                .build(),
        }
    }
}

node_to_inner!(ReduceMean);
