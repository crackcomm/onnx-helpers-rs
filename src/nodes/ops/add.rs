//! Add operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Add node.
pub struct Add {
    inner: Node,
}

impl Add {
    /// Creates new Add operation.
    #[inline(always)]
    pub fn new<Lhs: Into<String>, Rhs: Into<String>>(lhs: Lhs, rhs: Rhs) -> Self {
        Add {
            inner: builder::Node::new("Add").input(lhs).input(rhs).build(),
        }
    }
}

node_to_inner!(Add);
