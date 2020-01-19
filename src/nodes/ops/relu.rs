//! Relu operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Relu node.
pub struct Relu {
    inner: Node,
}

impl Relu {
    /// Creates new Relu operation.
    #[inline(always)]
    pub fn new<T: Into<String>>(input: T) -> Self {
        Relu {
            inner: builder::Node::new("Relu").input(input).build(),
        }
    }
}

node_to_inner!(Relu);
