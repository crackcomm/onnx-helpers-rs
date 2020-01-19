mod add;
mod and;
mod concat;
mod constant;
mod div;
mod equal;
mod greater;
mod less;
mod mul;
mod neg;
mod not;
mod or;
mod pow;
mod reduce_max;
mod reduce_mean;
mod reduce_min;
mod reduce_sum;
mod relu;
mod size;
mod sqrt;
mod sub;
mod tanh;

pub use self::add::*;
pub use self::and::*;
pub use self::concat::*;
pub use self::constant::*;
pub use self::div::*;
pub use self::equal::*;
pub use self::greater::*;
pub use self::less::*;
pub use self::mul::*;
pub use self::neg::*;
pub use self::not::*;
pub use self::or::*;
pub use self::pow::*;
pub use self::reduce_max::*;
pub use self::reduce_mean::*;
pub use self::reduce_min::*;
pub use self::reduce_sum::*;
pub use self::relu::*;
pub use self::size::*;
pub use self::sqrt::*;
pub use self::sub::*;
pub use self::tanh::*;

#[macro_export]
macro_rules! node_to_inner {
    (  $t: ty ) => {
        impl Into<crate::nodes::Node> for $t {
            #[inline(always)]
            fn into(self) -> crate::nodes::Node {
                self.inner
            }
        }

        impl Into<onnx_pb::NodeProto> for $t {
            #[inline(always)]
            fn into(self) -> onnx_pb::NodeProto {
                self.inner.into()
            }
        }

        impl Into<String> for &$t {
            #[inline(always)]
            fn into(self) -> String {
                (&self.inner).into()
            }
        }

        impl AsRef<crate::nodes::Node> for $t {
            #[inline(always)]
            #[inline(always)]
            fn as_ref(&self) -> &crate::nodes::Node {
                &self.inner
            }
        }
    };
}
