//! Onnx node helpers.

use onnx_pb::NodeProto;

use crate::builder;

/// Node wrapper.
#[derive(Clone, PartialEq)]
pub struct Node {
    inner: NodeProto,
}

impl Node {
    /// Returns protocol buffers representation.
    pub fn proto(&self) -> &NodeProto {
        &self.inner
    }

    fn output(&self) -> &String {
        if self.inner.op_type.is_empty() {
            &self.inner.name
        } else {
            self.inner.output.first().unwrap()
        }
    }
}

impl std::ops::Add for &Node {
    type Output = Node;

    fn add(self, rhs: &Node) -> Self::Output {
        builder::Node::new("Add")
            .input(self.output())
            .input(rhs.output())
            .build()
    }
}

impl From<NodeProto> for Node {
    fn from(inner: NodeProto) -> Self {
        Node { inner }
    }
}

impl Into<NodeProto> for Node {
    fn into(self) -> NodeProto {
        self.inner
    }
}
