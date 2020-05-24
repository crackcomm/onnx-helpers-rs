//! Reduce mean operation.

use onnx_pb::Axes;

use crate::{builder, node_to_inner, nodes::Node};

/// Reduce mean node.
pub struct ReduceMax {
    inner: Node,
}

impl ReduceMax {
    /// Creates new reduce mean operation.
    #[inline(always)]
    pub fn new<S: Into<String>, A: Into<Axes>>(input: S, axes: A, keepdims: bool) -> Self {
        ReduceMax {
            inner: builder::Node::new("ReduceMax")
                .input(input)
                .attribute("axes", axes.into())
                .attribute("keepdims", keepdims)
                .build(),
        }
    }
}

node_to_inner!(ReduceMax);
