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

    #[test]
    fn compare_with_py_output() {
        let from_python = ModelProto::decode(&read_buf("tests/model.onnx")).unwrap();
        // let x_input = make_tensor_value_info("X", DataType::Float, vec![1, 10], None);
        let x_input = builder::Value::new("X")
            .typed(DataType::Float)
            .shape(vec![1, 10]);
        let mean_reduce = builder::Node::new("ReduceMean")
            .input("X")
            .output("Z")
            .attribute("axes", vec![1i64]);
        let graph = builder::Graph::new("reduce-mean")
            .node(mean_reduce)
            .input(x_input);

        let model = builder::Model::new(graph).producer_name("reducer").build();
        assert_eq!(model, from_python);
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
