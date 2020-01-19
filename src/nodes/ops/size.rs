//! Size operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Size node.
pub struct Size {
    inner: Node,
}

impl Size {
    /// Creates new Size operation.
    #[inline(always)]
    pub fn new<T: Into<String>>(input: T) -> Self {
        Size {
            inner: builder::Node::new("Size").input(input).build(),
        }
    }
}

node_to_inner!(Size);
