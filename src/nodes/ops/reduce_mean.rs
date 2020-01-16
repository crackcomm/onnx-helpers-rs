//! Reduce mean operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Reduce mean node.
pub struct ReduceMean {
    inner: Node,
}

impl ReduceMean {
    /// Creates new reduce mean operation.
    pub fn new<S: Into<String>>(input: S) -> Self {
        ReduceMean {
            inner: builder::Node::new("ReduceMean").input(input).build(),
        }
    }
}

node_to_inner!(ReduceMean);
