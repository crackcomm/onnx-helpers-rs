//! Model encoding and decoding utilities.

use bytes::Buf;
use prost::Message;

use onnx_pb::ModelProto;

/// Decodes a protocol buffers message.
pub fn decode<T, B>(buf: B) -> Result<T, prost::DecodeError>
where
    B: Buf,
    T: Message + Sized + Default,
{
    let message = T::default();
    message.decode(buf)?;
    message
}
