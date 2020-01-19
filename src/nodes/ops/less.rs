//! Less operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Less node.
pub struct Less {
    inner: Node,
}

impl Less {
    /// Creates new Less operation.
    #[inline(always)]
    pub fn new<Lhs: Into<String>, Rhs: Into<String>>(lhs: Lhs, rhs: Rhs) -> Self {
        Less {
            inner: builder::Node::new("Less").input(lhs).input(rhs).build(),
        }
    }
}

node_to_inner!(Less);
