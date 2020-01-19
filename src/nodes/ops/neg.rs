//! Neg operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Neg node.
pub struct Neg {
    inner: Node,
}

impl Neg {
    /// Creates new Neg operation.
    #[inline(always)]
    pub fn new<T: Into<String>>(input: T) -> Self {
        Neg {
            inner: builder::Node::new("Neg").input(input).build(),
        }
    }
}

node_to_inner!(Neg);
