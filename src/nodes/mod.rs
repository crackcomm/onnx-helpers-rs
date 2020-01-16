//! Onnx node helpers.

pub mod ops;

use onnx_pb::NodeProto;

use crate::{attrs::Attribute, builder::Bag};

/// Node wrapper.
#[derive(Clone)]
pub struct Node {
    inner: NodeProto,
    pub(crate) bag: Option<Bag>,
}

impl Node {
    /// Returns node name.
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Returns protocol buffers representation.
    pub fn proto(&self) -> &NodeProto {
        &self.inner
    }

    /// Creates new square root operation.
    pub fn sqrt(&mut self) -> Node {
        let mut node: Node = ops::Sqrt::new(select_output(&self.inner)).into();
        maybe_bag_node(self.bag.as_mut(), &mut node);
        node
    }

    /// Creates new power operation.
    pub fn pow<T: Into<String>>(&mut self, power: T) -> Node {
        let mut node: Node = ops::Pow::new(select_output(&self.inner), power).into();
        maybe_bag_node(self.bag.as_mut(), &mut node);
        node
    }

    /// Creates new reduce sum operation.
    pub fn sum<A: Into<Attribute>>(&mut self, axes: A, keepdims: bool) -> Node {
        let mut node: Node = ops::ReduceSum::new(select_output(&self.inner), axes, keepdims).into();
        maybe_bag_node(self.bag.as_mut(), &mut node);
        node
    }

    /// Creates new reduce mean operation.
    pub fn mean<A: Into<Attribute>>(&mut self, axes: A, keepdims: bool) -> Node {
        let mut node: Node =
            ops::ReduceMean::new(select_output(&self.inner), axes, keepdims).into();
        maybe_bag_node(self.bag.as_mut(), &mut node);
        node
    }
}

impl<Rhs: AsRef<Node>> std::ops::Add<Rhs> for &mut Node {
    type Output = Node;

    fn add(self, rhs: Rhs) -> Self::Output {
        let mut node: Node = ops::Add::new(select_output(&self.inner), rhs.as_ref()).into();
        maybe_bag_node(self.bag.as_mut(), &mut node);
        node
    }
}

impl<Rhs: AsRef<Node>> std::ops::Sub<Rhs> for &mut Node {
    type Output = Node;

    fn sub(self, rhs: Rhs) -> Self::Output {
        let mut node: Node = ops::Sub::new(select_output(&self.inner), rhs.as_ref()).into();
        maybe_bag_node(self.bag.as_mut(), &mut node);
        node
    }
}

impl<Rhs: AsRef<Node>> std::ops::Mul<Rhs> for &mut Node {
    type Output = Node;

    fn mul(self, rhs: Rhs) -> Self::Output {
        let mut node: Node = ops::Mul::new(select_output(&self.inner), rhs.as_ref()).into();
        maybe_bag_node(self.bag.as_mut(), &mut node);
        node
    }
}

impl<Rhs: AsRef<Node>> std::ops::Div<Rhs> for &mut Node {
    type Output = Node;

    fn div(self, rhs: Rhs) -> Self::Output {
        let mut node: Node = ops::Div::new(select_output(&self.inner), rhs.as_ref()).into();
        maybe_bag_node(self.bag.as_mut(), &mut node);
        node
    }
}

impl std::ops::Neg for &mut Node {
    type Output = Node;

    fn neg(self) -> Self::Output {
        let mut node: Node = ops::Neg::new(select_output(&self.inner)).into();
        maybe_bag_node(self.bag.as_mut(), &mut node);
        node
    }
}

impl From<NodeProto> for Node {
    fn from(inner: NodeProto) -> Self {
        Node { bag: None, inner }
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

impl From<&Node> for String {
    fn from(node: &Node) -> String {
        select_output(&node.inner)
    }
}

impl From<&mut Node> for String {
    fn from(node: &mut Node) -> String {
        select_output(&node.inner)
    }
}

// impl Into<String> for &mut Node {
//     fn into(self) -> String {
//         select_output(self)
//     }
// }

// impl Into<String> for &mut Node {
//     fn into(self) -> String {
//         select_output(self)
//     }
// }

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
fn select_output(node: &NodeProto) -> String {
    if node.op_type.is_empty() {
        node.name.clone()
    } else {
        node.output.first().unwrap().to_owned()
    }
}

#[inline(always)]
pub(crate) fn maybe_bag_node(bag: Option<&mut Bag>, node: &mut Node) {
    if let Some(bag) = bag {
        node.bag = Some(bag.clone());
        bag.node(node.clone());
    }
}
