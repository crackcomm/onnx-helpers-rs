use prost::Message;

use onnx_helpers::prelude::*;
use onnx_pb::tensor_proto::DataType;

fn main() {
    // let input = Input::default();
    // let mean_reduce = MeanReduce::new(input)

    let x = builder::Node::default().name("X").build();
    let y = builder::Node::default().name("Y").build();

    let r = &x + &y;
    let r2 = &r + &x;
    let r3 = &r2 + &y;
    let r4 = &r3 + &r2;

    let x_input = builder::Value::new("X")
        .typed(DataType::Float)
        .shape(vec![1, 10]);
    let y_input = builder::Value::new("Y")
        .typed(DataType::Float)
        .shape(vec![1, 10]);
    let graph = builder::Graph::new("add")
        .node(r)
        .node(r2)
        .node(r3)
        .node(r4)
        .input(x_input)
        .input(y_input);

    let model = builder::Model::new(graph).producer_name("adder").build();

    let mut b = Vec::with_capacity(1024);
    model.encode(&mut b).unwrap();

    std::fs::write("onnx-model.onnx", b).unwrap();
}
