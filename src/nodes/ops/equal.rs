//! Equal operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Equal node.
pub struct Equal {
    inner: Node,
}

impl Equal {
    /// Creates new Equal operation.
    pub fn new<Lhs: Into<String>, Rhs: Into<String>>(lhs: Lhs, rhs: Rhs) -> Self {
        Equal {
            inner: builder::Node::new("Equal").input(lhs).input(rhs).build(),
        }
    }
}

node_to_inner!(Equal);
