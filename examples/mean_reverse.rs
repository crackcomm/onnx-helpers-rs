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
    let graph = graph.outputs(-(&x - mean) * two + x);
    let model = builder::Model::new(graph).producer_name("adder").build();

    let mut b = Vec::with_capacity(1024);
    model.encode(&mut b).unwrap();

    std::fs::write("onnx-model.onnx", b).unwrap();
}
