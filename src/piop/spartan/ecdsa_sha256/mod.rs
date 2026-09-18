//! One padded SHA-256 message and one P-256 verification, sharing a bit oracle.
//!
//! Linear rows bypass the nonlinear outer sumcheck and join its matrix claims
//! in one packed inner sumcheck. The terminal scaled claim is opened via BitZ.

mod codec;
mod inner_reduction;
mod proof;
mod relation;
mod security;
mod witness;

#[cfg(test)]
mod tests;

#[cfg(test)]
use crate::piop::spartan::SpartanField as _;
pub use proof::{Sha256EcdsaProof, commit_sha256_ecdsa, prove_sha256_ecdsa, verify_sha256_ecdsa};
pub use relation::{OuterMode, PreparedSha256Ecdsa, Sha256EcdsaStatement, prepare_sha256_ecdsa};
pub use security::{ChallengeSecurity, Sha256EcdsaSecurity};
pub use witness::{Sha256EcdsaWitness, generate_sha256_ecdsa_witness};

use crate::piop::spartan::bitz::SpartanBitzField as F;

#[cfg(test)]
#[cfg(test)]
use num_bigint::BigInt;
#[cfg(test)]
use num_traits::ToPrimitive;
use std::fmt::Display;

#[derive(Debug, thiserror::Error)]
#[error("SHA-256/ECDSA: {0}")]
pub struct Sha256EcdsaError(pub String);

pub(crate) type Result<T> = std::result::Result<T, Sha256EcdsaError>;
type Config = <F as crate::piop::spartan::SpartanField>::Config;

pub(crate) fn error(value: impl Display) -> Sha256EcdsaError {
    Sha256EcdsaError(value.to_string())
}

#[cfg(test)]
fn reduce_integer_mod_q(value: &BigInt, modulus: u128, cfg: &Config) -> F {
    let modulus = BigInt::from(modulus);
    let residue = ((value % &modulus) + &modulus) % modulus;
    F::from_with_cfg(
        residue
            .to_u128()
            .expect("residue is in 0..modulus, which fits u128"),
        cfg,
    )
}

impl From<super::sumcheck::SumcheckError> for Sha256EcdsaError {
    fn from(value: super::sumcheck::SumcheckError) -> Self {
        error(value)
    }
}
