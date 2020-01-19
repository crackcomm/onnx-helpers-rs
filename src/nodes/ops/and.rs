//! And operation.

use crate::{builder, node_to_inner, nodes::Node};

/// And node.
pub struct And {
    inner: Node,
}

impl And {
    /// Creates new And operation.
    #[inline(always)]
    pub fn new<Lhs: Into<String>, Rhs: Into<String>>(lhs: Lhs, rhs: Rhs) -> Self {
        And {
            inner: builder::Node::new("And").input(lhs).input(rhs).build(),
        }
    }
}

node_to_inner!(And);
