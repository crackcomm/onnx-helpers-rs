//! Model saving and opening.

use std::path::Path;

use prost::Message;

use onnx_pb::ModelProto;

/// File utils error.
#[derive(Debug)]
pub enum Error {
    /// IO error.
    Io(std::io::Error),

    /// Decode error.
    Decode(prost::DecodeError),

    /// Encode error.
    Encode(prost::EncodeError),
}

/// Opens model from a file.
pub fn open_model<P: AsRef<Path>>(path: P) -> Result<ModelProto, Error> {
    let body = std::fs::read(path).map_err(|e| Error::Io(e))?;
    ModelProto::decode(body.as_slice()).map_err(|e| Error::Decode(e))
}

/// Saves model to a file.
pub fn save_model<P: AsRef<Path>>(path: P, model: &ModelProto) -> Result<(), Error> {
    let mut body = Vec::new();
    model.encode(&mut body).map_err(|e| Error::Encode(e))?;
    std::fs::write(path, body).map_err(|e| Error::Io(e))?;
    Ok(())
}
