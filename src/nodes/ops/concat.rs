//! Concat operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Concat node.
pub struct Concat {
    inner: Node,
}

impl Concat {
    /// Creates new Concat operation.
    #[inline(always)]
    pub fn new<I>(axis: i64, inputs: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        Concat {
            inner: builder::Node::new("Concat")
                .inputs(inputs)
                .attribute("axis", axis)
                .build(),
        }
    }
}

node_to_inner!(Concat);
