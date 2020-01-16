use prost::Message;

use onnx_helpers::prelude::*;
use onnx_pb::tensor_proto::DataType;

fn main() {
    let mut graph = builder::Graph::new("add");

    let mut x = graph
        .input("X")
        .typed(DataType::Float)
        .shape(vec![1, 10])
        .node();

    let two = graph.constant(2.0f32);

    let mut mean = x.mean(vec![1i64], false);
    let mut x_sub_mean = &mut x - &mean;

    // Left part
    let mut x_sub_mean_neg = -&mut x_sub_mean;
    let mut x_sub_mean_neg_double = &mut x_sub_mean_neg * two;
    let mut mean_reverse = &mut x_sub_mean_neg_double + &x;

    let model = builder::Model::new(graph).producer_name("adder").build();

    let mut b = Vec::with_capacity(1024);
    model.encode(&mut b).unwrap();

    std::fs::write("onnx-model.onnx", b).unwrap();
}
