//! Not operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Not node.
pub struct Not {
    inner: Node,
}

impl Not {
    /// Creates new Not operation.
    #[inline(always)]
    pub fn new<T: Into<String>>(input: T) -> Self {
        Not {
            inner: builder::Node::new("Not").input(input).build(),
        }
    }
}

node_to_inner!(Not);
