//! Div operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Div node.
pub struct Div {
    inner: Node,
}

impl Div {
    /// Creates new Div operation.
    #[inline(always)]
    pub fn new<Lhs: Into<String>, Rhs: Into<String>>(lhs: Lhs, rhs: Rhs) -> Self {
        Div {
            inner: builder::Node::new("Div").input(lhs).input(rhs).build(),
        }
    }
}

node_to_inner!(Div);
