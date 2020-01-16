use prost::Message;

use onnx_helpers::prelude::*;
use onnx_pb::tensor_proto::DataType;

fn main() {
    // let input = Input::default();
    // let mean_reduce = MeanReduce::new(input)

    // let x = builder::Node::default().name("X").build();
    // let y = builder::Node::default().name("Y").build();

    // let one = Node::from(1.0f32);

    let mut graph = builder::Graph::new("add"); // .input(x_input).input(y_input);

    let mut x = graph
        .input("X")
        .typed(DataType::Float)
        .shape(vec![1, 10])
        .node();
    let mut y = graph
        .input("Y")
        .typed(DataType::Float)
        .shape(vec![1, 10])
        .node();

    let r = &mut x + &y;
    // println!("R: {}", r.proto().name);

    let model = builder::Model::new(graph.build())
        .producer_name("adder")
        .build();

    let mut b = Vec::with_capacity(1024);
    model.encode(&mut b).unwrap();

    std::fs::write("onnx-model.onnx", b).unwrap();
}
