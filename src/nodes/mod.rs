//! Onnx node helpers.

pub mod ops;

pub use self::ops::concat;

use std::cell::RefCell;
use std::rc::Rc;

use onnx_pb::NodeProto;

use crate::builder::Bag;

/// Node wrapper.
#[derive(Clone)]
pub struct Node {
    pub(crate) inner: Rc<RefCell<NodeProto>>,
    pub(crate) bag: Option<Bag>,
}

impl Node {
    /// Creates new node from proto.
    pub fn from_proto(inner: NodeProto) -> Self {
        Node {
            bag: None,
            inner: Rc::new(RefCell::new(inner)),
        }
    }

    /// Returns node name.
    pub fn name(&self) -> String {
        self.inner.borrow().name.clone()
    }

    /// Renames output names accordingly.
    pub fn with_name<N: Into<String>>(self, name: N) -> Self {
        let name = name.into();
        let mut bag = self.bag.clone();
        {
            let mut inner = self.inner.borrow_mut();
            inner
                .output
                .iter_mut()
                .enumerate()
                .for_each(|(index, output)| {
                    let name = format!("{}{}", name, index);
                    maybe_bag_rename(&mut bag, &output, &name);
                    *output = name;
                });
            maybe_bag_rename(&mut bag, &inner.name, &name);
            inner.name = name;
        }
        self
    }

    /// Creates new square root operation.
    pub fn sqrt(&self) -> Node {
        let mut node: Node = ops::Sqrt::new(self.select_output()).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new power operation.
    pub fn pow<T: Into<String>>(&self, power: T) -> Node {
        let mut node: Node = ops::Pow::new(self.select_output(), power).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new reduce sum operation.
    pub fn sum<A: Into<Axes>>(&self, axes: A, keepdims: bool) -> Node {
        let mut node: Node = ops::ReduceSum::new(self.select_output(), axes, keepdims).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new reduce max operation.
    pub fn max<A: Into<Axes>>(&self, axes: A, keepdims: bool) -> Node {
        let mut node: Node = ops::ReduceMax::new(self.select_output(), axes, keepdims).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new reduce mean operation.
    pub fn mean<A: Into<Axes>>(&self, axes: A, keepdims: bool) -> Node {
        let mut node: Node = ops::ReduceMean::new(self.select_output(), axes, keepdims).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new reduce min operation.
    pub fn min<A: Into<Axes>>(&self, axes: A, keepdims: bool) -> Node {
        let mut node: Node = ops::ReduceMin::new(self.select_output(), axes, keepdims).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new equal comparison operation.
    pub fn equal<Rhs: Into<String>>(&self, right: Rhs) -> Node {
        let mut node: Node = ops::Equal::new(self.select_output(), right).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new greater comparison operation.
    pub fn greater<Rhs: Into<String>>(&self, right: Rhs) -> Node {
        let mut node: Node = ops::Greater::new(self.select_output(), right).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new less comparison operation.
    pub fn less<Rhs: Into<String>>(&self, right: Rhs) -> Node {
        let mut node: Node = ops::Less::new(self.select_output(), right).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new logical and operation.
    pub fn and<Rhs: Into<String>>(&self, right: Rhs) -> Node {
        let mut node: Node = ops::And::new(self.select_output(), right).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new logical or operation.
    pub fn or<Rhs: Into<String>>(&self, right: Rhs) -> Node {
        let mut node: Node = ops::Or::new(self.select_output(), right).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new relu activation operation.
    pub fn relu(&self) -> Node {
        let mut node: Node = ops::Relu::new(self.select_output()).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new tanh activation operation.
    pub fn tanh(&self) -> Node {
        let mut node: Node = ops::Tanh::new(self.select_output()).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    /// Creates new size operation.
    pub fn size(&self) -> Node {
        let mut node: Node = ops::Size::new(self.select_output()).into();
        maybe_bag_node(self.bag.clone(), &mut node);
        node
    }

    #[inline]
    fn select_output(&self) -> String {
        let node = self.inner.borrow();
        if node.op_type.is_empty() {
            node.name.clone()
        } else {
            node.output.first().unwrap().to_owned()
        }
    }
}

#[macro_export]
macro_rules! impl_nodes_op {
    ( $t:ident, $k:ident, $f:ident ) => {
        impl<Rhs: AsRef<Node>> std::ops::$k<Rhs> for $t {
            type Output = Node;

            #[inline(always)]
            fn $f(self, rhs: Rhs) -> Self::Output {
                let mut node: Node = ops::$k::new(self.select_output(), rhs.as_ref()).into();
                maybe_bag_node(self.bag.clone(), &mut node);
                node
            }
        }

        impl<Rhs: AsRef<Node>> std::ops::$k<Rhs> for &$t {
            type Output = Node;

            #[inline(always)]
            fn $f(self, rhs: Rhs) -> Self::Output {
                let mut node: Node = ops::$k::new(self.select_output(), rhs.as_ref()).into();
                maybe_bag_node(self.bag.clone(), &mut node);
                node
            }
        }
    };
}

#[macro_export]
macro_rules! impl_node_op {
    ( $t:ident, $k:ident, $f:ident ) => {
        impl std::ops::$k for &$t {
            type Output = Node;

            #[inline(always)]
            fn $f(self) -> Self::Output {
                let mut node: Node = ops::$k::new(self.select_output()).into();
                maybe_bag_node(self.bag.clone(), &mut node);
                node
            }
        }

        impl std::ops::$k for $t {
            type Output = Node;

            #[inline(always)]
            fn $f(self) -> Self::Output {
                let mut node: Node = ops::$k::new(self.select_output()).into();
                maybe_bag_node(self.bag.clone(), &mut node);
                node
            }
        }
    };
}

impl_nodes_op!(Node, Add, add);
impl_nodes_op!(Node, Sub, sub);
impl_nodes_op!(Node, Mul, mul);
impl_nodes_op!(Node, Div, div);
impl_node_op!(Node, Neg, neg);
impl_node_op!(Node, Not, not);

impl From<NodeProto> for Node {
    fn from(inner: NodeProto) -> Self {
        Node::from_proto(inner)
    }
}

impl Into<NodeProto> for Node {
    fn into(self) -> NodeProto {
        self.inner.borrow().clone()
    }
}

impl From<Node> for String {
    fn from(node: Node) -> String {
        node.select_output()
    }
}

impl From<&Node> for String {
    fn from(node: &Node) -> String {
        node.select_output()
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

#[inline(always)]
pub(crate) fn maybe_bag_node(bag: Option<Bag>, node: &mut Node) {
    if let Some(mut bag) = bag {
        node.bag = Some(bag.clone());
        bag.node(node.inner.clone());
    }
}

#[inline(always)]
pub(crate) fn maybe_bag_rename(bag: &mut Option<Bag>, name: &str, new_name: &str) {
    if let Some(bag) = bag.as_mut() {
        bag.rename(name, new_name);
    }
}
