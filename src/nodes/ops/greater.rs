//! Greater operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Greater node.
pub struct Greater {
    inner: Node,
}

impl Greater {
    /// Creates new Greater operation.
    pub fn new<Lhs: Into<String>, Rhs: Into<String>>(lhs: Lhs, rhs: Rhs) -> Self {
        Greater {
            inner: builder::Node::new("Greater").input(lhs).input(rhs).build(),
        }
    }
}

node_to_inner!(Greater);
