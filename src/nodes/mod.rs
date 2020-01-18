//! Onnx node helpers.

pub mod ops;

use onnx_pb::NodeProto;

use crate::builder::Bag;

/// Node wrapper.
#[derive(Clone)]
pub struct Node {
    inner: NodeProto,
    pub(crate) bag: Option<Bag>,
}

impl Node {
    /// Creates new node from proto.
    pub fn from_proto(inner: NodeProto) -> Self {
        Node { bag: None, inner }
    }

    /// Returns node name.
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Returns protocol buffers representation.
    pub fn proto(&self) -> &NodeProto {
        &self.inner
    }

    /// Creates new square root operation.
    pub fn sqrt(&self) -> Node {
        let mut node: Node = ops::Sqrt::new(select_output(&self.inner)).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new power operation.
    pub fn pow<T: Into<String>>(&self, power: T) -> Node {
        let mut node: Node = ops::Pow::new(select_output(&self.inner), power).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new reduce sum operation.
    pub fn sum<A: Into<Axes>>(&self, axes: A, keepdims: bool) -> Node {
        let mut node: Node = ops::ReduceSum::new(select_output(&self.inner), axes, keepdims).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new reduce mean operation.
    pub fn mean<A: Into<Axes>>(&self, axes: A, keepdims: bool) -> Node {
        let mut node: Node =
            ops::ReduceMean::new(select_output(&self.inner), axes, keepdims).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }
}

impl<Rhs: AsRef<Node>> std::ops::Add<Rhs> for &Node {
    type Output = Node;

    fn add(self, rhs: Rhs) -> Self::Output {
        let mut node: Node = ops::Add::new(select_output(&self.inner), rhs.as_ref()).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }
}

impl<Rhs: AsRef<Node>> std::ops::Add<Rhs> for Node {
    type Output = Node;

    fn add(self, rhs: Rhs) -> Self::Output {
        let mut node: Node = ops::Add::new(select_output(&self.inner), rhs.as_ref()).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }
}

impl<Rhs: AsRef<Node>> std::ops::Sub<Rhs> for &Node {
    type Output = Node;

    fn sub(self, rhs: Rhs) -> Self::Output {
        let mut node: Node = ops::Sub::new(select_output(&self.inner), rhs.as_ref()).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }
}

impl<Rhs: AsRef<Node>> std::ops::Sub<Rhs> for Node {
    type Output = Node;

    fn sub(self, rhs: Rhs) -> Self::Output {
        let mut node: Node = ops::Sub::new(select_output(&self.inner), rhs.as_ref()).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }
}

impl<Rhs: AsRef<Node>> std::ops::Mul<Rhs> for &Node {
    type Output = Node;

    fn mul(self, rhs: Rhs) -> Self::Output {
        let mut node: Node = ops::Mul::new(select_output(&self.inner), rhs.as_ref()).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }
}

impl<Rhs: AsRef<Node>> std::ops::Mul<Rhs> for Node {
    type Output = Node;

    fn mul(self, rhs: Rhs) -> Self::Output {
        let mut node: Node = ops::Mul::new(select_output(&self.inner), rhs.as_ref()).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }
}

impl<Rhs: AsRef<Node>> std::ops::Div<Rhs> for &Node {
    type Output = Node;

    fn div(self, rhs: Rhs) -> Self::Output {
        let mut node: Node = ops::Div::new(select_output(&self.inner), rhs.as_ref()).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }
}

impl<Rhs: AsRef<Node>> std::ops::Div<Rhs> for Node {
    type Output = Node;

    fn div(self, rhs: Rhs) -> Self::Output {
        let mut node: Node = ops::Div::new(select_output(&self.inner), rhs.as_ref()).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }
}

impl std::ops::Neg for &Node {
    type Output = Node;

    fn neg(self) -> Self::Output {
        let mut node: Node = ops::Neg::new(select_output(&self.inner)).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }
}

impl std::ops::Neg for Node {
    type Output = Node;

    fn neg(self) -> Self::Output {
        let mut node: Node = ops::Neg::new(select_output(&self.inner)).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }
}

impl From<NodeProto> for Node {
    fn from(inner: NodeProto) -> Self {
        Node::from_proto(inner)
    }
}

impl Into<NodeProto> for Node {
    fn into(self) -> NodeProto {
        self.inner
    }
}

impl From<Node> for String {
    fn from(node: Node) -> String {
        select_output(&node.inner)
    }
}

impl From<&Node> for String {
    fn from(node: &Node) -> String {
        select_output(&node.inner)
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

/// Axes helpers struct.
pub struct Axes(pub Vec<i64>);

impl From<i64> for Axes {
    fn from(axes: i64) -> Self {
        Axes(vec![axes])
    }
}

impl From<Vec<i64>> for Axes {
    fn from(axes: Vec<i64>) -> Self {
        Axes(axes)
    }
}

#[inline]
fn select_output(node: &NodeProto) -> String {
    if node.op_type.is_empty() {
        node.name.clone()
    } else {
        node.output.first().unwrap().to_owned()
    }
}

#[inline(always)]
pub(crate) fn maybe_bag_node(bag: Option<Bag>, node: &mut Node) {
    if let Some(mut bag) = bag {
        node.bag = Some(bag.clone());
        bag.node(node.clone());
    }
}
