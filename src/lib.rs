//! ONNX model construction helpers.

pub mod attrs;
pub mod builder;
pub mod files;
pub mod nodes;
// pub mod proto;

pub use self::files::{open_model, save_model};

pub mod prelude {
    pub use crate::attrs::*;
    pub use crate::builder;
    pub use crate::files::*;
    pub use crate::nodes::ops::*;
    pub use crate::nodes::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    use prost::Message;

    use onnx_pb::{tensor_proto::DataType, ModelProto};
    use onnx_shape_inference::shape_inference;

    #[test]
    fn compare_with_prev_output() {
        let prev_output =
            ModelProto::decode(read_buf("tests/mean-reverse.onnx").as_slice()).unwrap();
        let mut graph = builder::Graph::new("reverse");
        let x = graph.input("X").typed(DataType::Float).dim(1).dim(6).node();
        let two = graph.constant(2.0f32);
        let graph = graph.outputs(-(&x - x.mean(1, true)) * two + x);
        let model = graph.model().build();
        let inferred = shape_inference(&model).unwrap();

        assert_eq!(inferred, prev_output);
    }

    fn read_buf<P: AsRef<std::path::Path>>(path: P) -> Vec<u8> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).unwrap();
        let mut buffer = Vec::new();
        // read the whole file
        file.read_to_end(&mut buffer).unwrap();
        buffer
    }
}
