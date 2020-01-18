use onnx_helpers::prelude::*;
use onnx_pb::tensor_proto::DataType;

fn main() {
    let mut graph = builder::Graph::new("add");
    let x = graph.input("X").typed(DataType::Float).dim(1).dim(6).node();
    let two = graph.constant(2.0f32);
    let graph = graph.outputs(-(&x - x.mean(1, true)) * two + x);
    let model = graph.model().build();
    save_model("mean-reverse.onnx", &model).unwrap();
}
