//! Nodes bag.

use std::cell::RefCell;
use std::rc::Rc;

use onnx_pb::ValueInfoProto;

use crate::nodes;

/// Bag marker.
#[derive(Copy, Clone)]
pub(crate) enum Marker {
    Input,
    Output,
}

/// Nodes bag.
#[derive(Clone, Default)]
pub(crate) struct Bag {
    inner: Rc<RefCell<BagInner>>,
}

impl Bag {
    pub fn nodes(&self) -> Vec<nodes::Node> {
        self.inner.borrow().nodes.clone()
    }

    pub fn inputs(&self) -> Vec<ValueInfoProto> {
        self.inner.borrow().inputs.clone()
    }

    pub fn outputs(&self) -> Vec<ValueInfoProto> {
        self.inner.borrow().outputs.clone()
    }

    pub fn value(&mut self, value: ValueInfoProto, marker: Marker) {
        match marker {
            Marker::Input => self.inner.borrow_mut().inputs.push(value),
            Marker::Output => self.inner.borrow_mut().outputs.push(value),
        }
    }

    pub fn node<T: Into<nodes::Node>>(&mut self, node: T) {
        let node = node.into();
        self.inner.borrow_mut().nodes.push(node)
    }
}

#[derive(Clone, Default)]
struct BagInner {
    nodes: Vec<nodes::Node>,
    inputs: Vec<ValueInfoProto>,
    outputs: Vec<ValueInfoProto>,
}
