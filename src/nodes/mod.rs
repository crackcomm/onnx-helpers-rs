//! Onnx node helpers.

pub mod ops;

use onnx_pb::NodeProto;

use crate::attrs::Attribute;

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

    /// Creates new square root operation.
    pub fn sqrt(&self) -> Node {
        ops::Sqrt::new(self).into()
    }

    /// Creates new power operation.
    pub fn pow<T: Into<String>>(&self, power: T) -> Node {
        ops::Pow::new(self, power).into()
    }

    /// Creates new reduce sum operation.
    pub fn sum<A: Into<Attribute>>(&self, axes: A, keepdims: bool) -> Node {
        ops::ReduceSum::new(self, axes, keepdims).into()
    }
}

impl<Rhs: AsRef<Node>> std::ops::Add<Rhs> for &Node {
    type Output = Node;

    fn add(self, rhs: Rhs) -> Self::Output {
        ops::Add::new(self, rhs.as_ref()).into()
    }
}

impl<Rhs: AsRef<Node>> std::ops::Sub<Rhs> for &Node {
    type Output = Node;

    fn sub(self, rhs: Rhs) -> Self::Output {
        ops::Sub::new(self, rhs.as_ref()).into()
    }
}

impl<Rhs: AsRef<Node>> std::ops::Mul<Rhs> for &Node {
    type Output = Node;

    fn mul(self, rhs: Rhs) -> Self::Output {
        ops::Mul::new(self, rhs.as_ref()).into()
    }
}

impl<Rhs: AsRef<Node>> std::ops::Div<Rhs> for &Node {
    type Output = Node;

    fn div(self, rhs: Rhs) -> Self::Output {
        ops::Div::new(self, rhs.as_ref()).into()
    }
}

impl std::ops::Neg for &Node {
    type Output = Node;

    fn neg(self) -> Self::Output {
        ops::Neg::new(self).into()
    }
}

impl From<NodeProto> for Node {
    fn from(inner: NodeProto) -> Self {
        Node { inner }
    }
}

impl From<f32> for Node {
    fn from(value: f32) -> Self {
        ops::Constant::new(format!("C_{:.2}", value).replace('.', "_"), vec![value]).into()
    }
}

impl Into<NodeProto> for Node {
    fn into(self) -> NodeProto {
        self.inner
    }
}

impl Into<String> for &Node {
    fn into(self) -> String {
        select_output(self)
    }
}

impl AsRef<NodeProto> for Node {
    #[inline(always)]
    fn as_ref(&self) -> &NodeProto {
        &self.inner
    }
}

impl AsRef<Node> for Node {
    #[inline(always)]
    fn as_ref(&self) -> &Node {
        &self
    }
}

#[inline]
fn select_output<T: AsRef<NodeProto>>(node: T) -> String {
    let node = node.as_ref();
    if node.op_type.is_empty() {
        node.name.clone()
    } else {
        node.output.first().unwrap().to_owned()
    }
}

/// Input node helper.
#[derive(Default)]
pub struct Input;
