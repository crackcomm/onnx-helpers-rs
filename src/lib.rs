//! ONNX model construction helpers.

pub mod builder;
pub mod nodes;

pub mod prelude {
    pub use crate::builder;
    pub use crate::nodes::ops::*;
    pub use crate::nodes::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    use onnx_pb::{open_model, tensor_proto::DataType};
    use onnx_shape_inference::shape_inference;

    #[test]
    fn compare_with_prev_output() {
        let prev_output = open_model("tests/mean-reverse.onnx").unwrap();
        let mut graph = builder::Graph::new("reverse");
        let x = graph.input("X").typed(DataType::Float).dim(1).dim(6).node();
        let two = graph.constant("two", 2.0f32);
        let graph = graph.outputs_typed(-(&x - x.mean(1, true)) * two + x, DataType::Float);
        let model = graph.model().build();
        let inferred = shape_inference(&model).unwrap();

        assert_eq!(inferred, prev_output);
    }
}
