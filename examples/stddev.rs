use onnx_helpers::prelude::*;
use onnx_pb::{save_model, tensor_proto::DataType};
use onnx_shape_inference::shape_inference;

fn main() {
    let mut graph = builder::Graph::new("stddev");
    let x = graph.input("X").typed(DataType::Float).dim(1).dim(6).node();
    let two = graph.constant("two", 2.0f32);
    let std = (&x - x.mean(1, true)).abs().pow(two).mean(1, true).sqrt();
    let graph = graph.outputs_typed(std.with_name("stddev"), DataType::Float);
    let model = shape_inference(&graph.model().build()).unwrap();
    let model = shape_inference(&model).unwrap();
    let model = shape_inference(&model).unwrap();
    let model = shape_inference(&model).unwrap();
    save_model("stddev.onnx", &model).unwrap();
}
