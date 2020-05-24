use onnx_helpers::prelude::*;
use onnx_pb::{save_model, tensor_proto::DataType};

fn main() {
    let mut graph = builder::Graph::new("add");

    let x = graph
        .input("X")
        .typed(DataType::Float)
        .dim(1)
        .dim(10)
        .node();
    let y = graph
        .input("Y")
        .typed(DataType::Float)
        .dim(1)
        .dim(10)
        .node();

    let z = (x + y).with_name("Z");

    let model = graph.outputs(z).model().build();

    save_model("add.onnx", &model).unwrap();
}
