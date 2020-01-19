# onnx-helpers

[![Documentation](https://docs.rs/onnx-helpers/badge.svg)](https://docs.rs/onnx-helpers/)
[![Crate](https://img.shields.io/crates/v/onnx-helpers.svg)](https://crates.io/crates/onnx-helpers)

ONNX graph construction helpers.

## Usage

```Toml
[dependencies]
onnx-helpers = "^1.3.0"
```

## Example

```Rust
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
```

## Credits

Based on [onnx-rs](https://github.com/nhynes/onnx-rs/).

## License

MIT license same as [ONNX](https://github.com/onnx/onnx).
