use onnx_helpers::prelude::*;
use onnx_pb::{save_model, tensor_proto::DataType};
use onnx_shape_inference::shape_inference;

fn main() {
    let mut graph = builder::Graph::new("concat");
    let x = graph.input("X").typed(DataType::Float).dim(1).dim(6).node();
    let two = graph.constant("two", 2.0f32);
    let mean_reverse = -(&x - x.mean(1, true)) * two + &x;
    let concat = graph.concat(0, vec![x, mean_reverse]).with_name("concat");
    let graph = graph.outputs_typed(concat, DataType::Float);
    let model = shape_inference(&graph.model().build()).unwrap();
    save_model("concat.onnx", &model).unwrap();
}
