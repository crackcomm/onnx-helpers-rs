//! Pow operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Pow node.
pub struct Pow {
    inner: Node,
}

impl Pow {
    /// Creates new Pow operation.
    #[inline(always)]
    pub fn new<Lhs: Into<String>, Rhs: Into<String>>(lhs: Lhs, rhs: Rhs) -> Self {
        Pow {
            inner: builder::Node::new("Pow").input(lhs).input(rhs).build(),
        }
    }
}

node_to_inner!(Pow);
