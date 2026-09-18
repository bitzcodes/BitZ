//! Binary and prime field arithmetic.
#![doc = include_str!("type_contracts.md")]

pub mod batch;
pub mod binary;
pub use batch::{DotProductKernels, SumcheckKernels};
#[cfg(feature = "spongefish")]
mod codec;
pub mod ct;
pub mod encoding;
pub mod gf128;
pub mod integer;
pub mod modular;
pub mod preparation;
pub mod prime;
pub use prime::PreparedSignedProjection;
pub mod traits;

pub use binary::{
    B127, B127Ops, B127Product, F2, F2Ops, Gf8, Gf8Ops, Gf8Product, Gf128Ops, Gf128PreparedAcc,
    Gf128Product, PreparedB127Mul, PreparedGf128Mul, PreparedGf128Projection, XorAccumulator,
};
pub use binary::{F2Poly, F2PolyOps, F2PolyProduct};
pub use ct::{Bit, CtEq, CtMask, CtOrd, CtSelect, CtValue};
pub use encoding::{CanonicalCodec, DecodeError};
pub use gf128::{Gf128, Wide256};
pub use integer::{
    IntegerOps, Uint, UintAccumulator, UintProduct, UintRef, WrappingOps, Z, ZAccumulator,
    ZProduct, ZRef,
};
pub use modular::{ContextError, ModRingCtx, Residue};
pub use preparation::{
    FixedBasePow, PreparedDivisor, PreparedIntegerProjection, PreparedOddInverse, PreparedProducts,
    PublicProductBounds, ShapeError,
};
pub use prime::*;
pub use traits::*;
