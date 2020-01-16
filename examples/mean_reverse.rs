use prost::Message;

use onnx_helpers::prelude::*;
use onnx_pb::tensor_proto::DataType;

fn main() {
    let mut graph = builder::Graph::new("add");

    let mut x = graph
        .input("X")
        .typed(DataType::Float)
        .shape(vec![1, 6])
        .node();

    let two = graph.constant(2.0f32);

    let mean = x.mean(vec![1i64], true);
    let mean_diff = -(&x - mean);
    let double_mean_diff = mean_diff * two;
    let graph = graph.outputs(double_mean_diff + x);
    let model = builder::Model::new(graph).producer_name("adder").build();

    let mut b = Vec::with_capacity(1024);
    model.encode(&mut b).unwrap();

    std::fs::write("onnx-model.onnx", b).unwrap();
}
