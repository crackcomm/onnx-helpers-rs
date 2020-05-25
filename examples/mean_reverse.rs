use onnx_helpers::prelude::*;
use onnx_pb::{save_model, tensor_proto::DataType};

fn main() {
    let mut graph = builder::Graph::new("reverse");
    let x = graph.input("X").typed(DataType::Float).dim(1).dim(6).node();
    let two = graph.constant("two", 2.0f32);
    let out = -(&x - x.mean(1, true)) * two + x;
    let graph = graph.outputs_typed(out, DataType::Float);
    let model = graph.model().build();
    save_model("mean-reverse.onnx", &model).unwrap();
}
