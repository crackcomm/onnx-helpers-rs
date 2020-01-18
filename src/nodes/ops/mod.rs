mod add;
mod constant;
mod div;
mod mul;
mod neg;
mod pow;
mod reduce_mean;
mod reduce_sum;
mod sqrt;
mod sub;

pub use self::add::*;
pub use self::constant::*;
pub use self::div::*;
pub use self::mul::*;
pub use self::neg::*;
pub use self::pow::*;
pub use self::reduce_mean::*;
pub use self::reduce_sum::*;
pub use self::sqrt::*;
pub use self::sub::*;

#[macro_export]
macro_rules! node_to_inner {
    (  $t: ty ) => {
        impl Into<crate::nodes::Node> for $t {
            fn into(self) -> crate::nodes::Node {
                self.inner
            }
        }

        impl Into<onnx_pb::NodeProto> for $t {
            fn into(self) -> onnx_pb::NodeProto {
                self.inner.into()
            }
        }

        impl Into<String> for &$t {
            fn into(self) -> String {
                (&self.inner).into()
            }
        }

        impl AsRef<crate::nodes::Node> for $t {
            #[inline(always)]
            fn as_ref(&self) -> &crate::nodes::Node {
                &self.inner
            }
        }
    };
}
