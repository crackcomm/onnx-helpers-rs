//! Concat operation.

use crate::{builder, node_to_inner, nodes::Node};

/// Concat node.
pub struct Concat {
    inner: Node,
}

impl Concat {
    /// Creates new Concat operation.
    #[inline(always)]
    pub fn new<I>(inputs: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        Concat {
            inner: builder::Node::new("Concat").inputs(inputs).build(),
        }
    }
}

/// Creates new Concat operation.
#[inline(always)]
pub fn concat<I>(inputs: I) -> Concat
where
    I: IntoIterator,
    I::Item: Into<String>,
{
    Concat {
        inner: builder::Node::new("Concat").inputs(inputs).build(),
    }
}

node_to_inner!(Concat);
