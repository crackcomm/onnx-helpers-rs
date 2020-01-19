//! Abs operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Abs node.
pub struct Abs {
    inner: Node,
}

impl Abs {
    /// Creates new Abs operation.
    #[inline(always)]
    pub fn new<T: Into<String>>(input: T) -> Self {
        Abs {
            inner: builder::Node::new("Abs").input(input).build(),
        }
    }
}

node_to_inner!(Abs);
