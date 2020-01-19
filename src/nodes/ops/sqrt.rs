//! Sqrt operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Sqrt node.
pub struct Sqrt {
    inner: Node,
}

impl Sqrt {
    /// Creates new Sqrt operation.
    #[inline(always)]
    pub fn new<T: Into<String>>(input: T) -> Self {
        Sqrt {
            inner: builder::Node::new("Sqrt").input(input).build(),
        }
    }
}

node_to_inner!(Sqrt);
