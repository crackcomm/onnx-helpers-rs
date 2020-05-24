//! Reduce mean operation.

use onnx_pb::Axes;

use crate::{builder, node_to_inner, nodes::Node};

/// Reduce mean node.
pub struct ReduceMean {
    inner: Node,
}

impl ReduceMean {
    /// Creates new reduce mean operation.
    #[inline(always)]
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
