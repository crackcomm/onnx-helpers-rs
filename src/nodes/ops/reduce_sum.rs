//! Reduce sum operation.

use crate::{
    builder, node_to_inner,
    nodes::{Axes, Node},
};

/// Reduce sum node.
pub struct ReduceSum {
    inner: Node,
}

impl ReduceSum {
    /// Creates new reduce sum operation.
    #[inline(always)]
    pub fn new<S: Into<String>, A: Into<Axes>>(input: S, axes: A, keepdims: bool) -> Self {
        ReduceSum {
            inner: builder::Node::new("ReduceSum")
                .input(input)
                .attribute("axes", axes.into())
                .attribute("keepdims", keepdims)
                .build(),
        }
    }
}

node_to_inner!(ReduceSum);
