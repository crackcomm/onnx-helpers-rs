//! Tanh operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Tanh node.
pub struct Tanh {
    inner: Node,
}

impl Tanh {
    /// Creates new Tanh operation.
    #[inline(always)]
    pub fn new<T: Into<String>>(input: T) -> Self {
        Tanh {
            inner: builder::Node::new("Tanh").input(input).build(),
        }
    }
}

node_to_inner!(Tanh);
