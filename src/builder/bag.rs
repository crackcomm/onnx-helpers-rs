//! Nodes bag.

use std::cell::RefCell;
use std::rc::Rc;

use onnx_pb::{NodeProto, ValueInfoProto};

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
    pub fn nodes(&self) -> Vec<NodeProto> {
        self.inner
            .borrow()
            .nodes
            .iter()
            .map(|n| n.borrow().clone())
            .collect()
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

    pub fn node(&mut self, node: Rc<RefCell<NodeProto>>) {
        self.inner.borrow_mut().nodes.push(node)
    }

    pub fn rename(&mut self, name: &str, new_name: &str) {
        self.inner.borrow_mut().rename(name, new_name)
    }
}

#[derive(Clone, Default)]
struct BagInner {
    nodes: Vec<Rc<RefCell<NodeProto>>>,
    inputs: Vec<ValueInfoProto>,
    outputs: Vec<ValueInfoProto>,
}

impl BagInner {
    pub fn rename(&mut self, name: &str, new_name: &str) {
        for node in self.nodes.iter_mut() {
            match node.try_borrow_mut() {
                Ok(mut node) => {
                    if node.name == name {
                        node.name = new_name.to_owned();
                    }
                }
                Err(_) => {
                    // it is the node we're changing right now
                }
            }
        }
        for input in self.inputs.iter_mut() {
            if input.name == name {
                input.name = new_name.to_owned();
            }
        }
        for node in self.outputs.iter_mut() {
            if node.name == name {
                node.name = new_name.to_owned();
            }
        }
    }
}
