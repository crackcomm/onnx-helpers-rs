//! Or operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Or node.
pub struct Or {
    inner: Node,
}

impl Or {
    /// Creates new Or operation.
    #[inline(always)]
    pub fn new<Lhs: Into<String>, Rhs: Into<String>>(lhs: Lhs, rhs: Rhs) -> Self {
        Or {
            inner: builder::Node::new("Or").input(lhs).input(rhs).build(),
        }
    }
}

node_to_inner!(Or);
