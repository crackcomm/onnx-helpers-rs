use onnx_helpers::prelude::*;
use onnx_pb::tensor_proto::DataType;
use onnx_shape_inference::shape_inference;

fn main() {
    let mut graph = builder::Graph::new("reverse");
    let x = graph.input("X").typed(DataType::Float).dim(1).dim(6).node();
    let two = graph.constant("two", 2.0f32);
    let graph = graph.outputs(-(&x - x.mean(1, true)) * two + x);
    let model = shape_inference(&graph.model().build()).unwrap();
    save_model("mean-reverse.onnx", &model).unwrap();
}
