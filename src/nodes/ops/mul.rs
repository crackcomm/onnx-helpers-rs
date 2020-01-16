//! Mul operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Mul node.
pub struct Mul {
    inner: Node,
}

impl Mul {
    /// Creates new Mul operation.
    pub fn new<Lhs: Into<String>, Rhs: Into<String>>(lhs: Lhs, rhs: Rhs) -> Self {
        Mul {
            inner: builder::Node::new("Mul").input(lhs).input(rhs).build(),
        }
    }
}

node_to_inner!(Mul);
