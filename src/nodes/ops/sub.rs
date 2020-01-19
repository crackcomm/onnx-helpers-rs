//! Sub operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Sub node.
pub struct Sub {
    inner: Node,
}

impl Sub {
    /// Creates new Sub operation.
    #[inline(always)]
    pub fn new<Lhs: Into<String>, Rhs: Into<String>>(lhs: Lhs, rhs: Rhs) -> Self {
        Sub {
            inner: builder::Node::new("Sub").input(lhs).input(rhs).build(),
        }
    }
}

node_to_inner!(Sub);
