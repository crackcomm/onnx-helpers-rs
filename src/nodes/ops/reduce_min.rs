//! Reduce mean operation.

use crate::{
    builder, node_to_inner,
    nodes::{Axes, Node},
};

/// Reduce mean node.
pub struct ReduceMin {
    inner: Node,
}

impl ReduceMin {
    /// Creates new reduce mean operation.
    #[inline(always)]
    pub fn new<S: Into<String>, A: Into<Axes>>(input: S, axes: A, keepdims: bool) -> Self {
        ReduceMin {
            inner: builder::Node::new("ReduceMin")
                .input(input)
                .attribute("axes", axes.into())
                .attribute("keepdims", keepdims)
                .build(),
        }
    }
}

node_to_inner!(ReduceMin);
