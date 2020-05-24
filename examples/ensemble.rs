use onnx_helpers::prelude::*;
use onnx_pb::{save_model, tensor_proto::DataType};
use onnx_shape_inference::shape_inference;

fn main() {
    let mut graph = builder::Graph::new("stddev");
    let x = graph.input("X").typed(DataType::Float).dim(1).dim(6).node();
    let std = stddev(&mut graph, &x);
    let mrev = mean_reverse(&mut graph, &x);
    let graph = graph
        .outputs_typed(std.with_name("stddev"), DataType::Float)
        .outputs_typed(mrev.with_name("mean_reverse"), DataType::Float);
    let model = shape_inference(&graph.model().build()).unwrap();
    save_model("ensemble.onnx", &model).unwrap();
}

fn stddev(graph: &mut builder::Graph, x: &Node) -> Node {
    let two = graph.constant("two", 2.0f32);
    (x - x.mean(1, true)).abs().pow(two).mean(1, true).sqrt()
}

fn mean_reverse(graph: &mut builder::Graph, x: &Node) -> Node {
    let two = graph.constant("two", 2.0f32);
    -(x - x.mean(1, true)) * two + x
}
