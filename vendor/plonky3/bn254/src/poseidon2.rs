//! Diffusion matrix for Bn254
//!
//! Reference: <https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_bn256.rs>.

use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use p3_poseidon2::{
    ExternalLayer, ExternalLayerConstants, ExternalLayerConstructor, HLMDSMat4, InternalLayer,
    InternalLayerConstructor, Poseidon2, add_rc_and_sbox_generic, external_initial_permute_state,
    external_terminal_permute_state, internal_permute_state,
};

use crate::Bn254;

/// S-box degree for BN254 Poseidon2.
///
/// Since `p - 1` is divisible by 2 and 3, the smallest valid exponent
/// satisfying `gcd(α, p - 1) = 1` is 5.
pub const BN254_S_BOX_DEGREE: u64 = 5;

/// Half the number of full rounds for BN254 Poseidon2.
///
/// Full rounds run at the beginning and at the end of the schedule.
/// This constant is the per-side count, so the total is `R_F = 8`.
///
/// Of those 8 rounds:
/// - 6 are required by the wide-trail differential bound,
/// - 2 are kept as a statistical security margin.
///
/// # Reference
///
/// Poseidon2 paper, Table 1, instance `(n, t, d) = (256, 3, 5)`:
/// <https://eprint.iacr.org/2023/323>.
pub const BN254_POSEIDON2_HALF_FULL_ROUNDS: usize = 4;

/// Number of partial rounds for BN254 Poseidon2 at state width 3.
///
/// Partial rounds apply the S-box to a single state element and carry
/// most of the algebraic security of the permutation.
///
/// # Reference
///
/// Poseidon2 paper, Table 1, instance `(n, t, d) = (256, 3, 5)` at the
/// 128-bit security level: <https://eprint.iacr.org/2023/323>.
pub const BN254_POSEIDON2_PARTIAL_ROUNDS_3: usize = 56;

/// An implementation of the Poseidon2 hash function for the Bn254Fr field.
///
/// It acts on arrays of the form `[Bn254Fr; WIDTH]`.
pub type Poseidon2Bn254<const WIDTH: usize> = Poseidon2<
    Bn254,
    Poseidon2ExternalLayerBn254<WIDTH>,
    Poseidon2InternalLayerBn254,
    WIDTH,
    BN254_S_BOX_DEGREE,
>;

/// Currently we only support a single width for Poseidon2 BN254.
const BN254_WIDTH: usize = 3;

#[derive(Debug, Clone, Default)]
pub struct Poseidon2InternalLayerBn254 {
    internal_constants: Vec<Bn254>,
}

impl InternalLayerConstructor<Bn254> for Poseidon2InternalLayerBn254 {
    fn new_from_constants(internal_constants: Vec<Bn254>) -> Self {
        Self { internal_constants }
    }
}

/// A faster version of `matmul_internal` making use of the fact that
/// the internal matrix is equal to:
/// ```ignore
///                             [2, 1, 1]
///     1 + Diag([1, 1, 2]) =   [1, 2, 1]
///                             [1, 1, 3]
/// ```
fn bn254_matmul_internal(state: &mut [Bn254; 3]) {
    // We bracket in this way as the s-box is applied to state[0] so this lets us
    // begin this computation before the s-box finishes.
    let sum = state[0] + (state[1] + state[2]);

    state[0] += sum;
    state[1] += sum;
    state[2] = state[2].double() + sum;
}

impl InternalLayer<Bn254, BN254_WIDTH, BN254_S_BOX_DEGREE> for Poseidon2InternalLayerBn254 {
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [Bn254; BN254_WIDTH]) {
        internal_permute_state(state, bn254_matmul_internal, &self.internal_constants);
    }
}

pub type Poseidon2ExternalLayerBn254<const WIDTH: usize> = ExternalLayerConstants<Bn254, WIDTH>;

impl<const WIDTH: usize> ExternalLayerConstructor<Bn254, WIDTH>
    for Poseidon2ExternalLayerBn254<WIDTH>
{
    fn new_from_constants(external_constants: Self) -> Self {
        external_constants
    }
}

impl<const WIDTH: usize> ExternalLayer<Bn254, WIDTH, BN254_S_BOX_DEGREE>
    for Poseidon2ExternalLayerBn254<WIDTH>
{
    /// Perform the initial external layers of the Poseidon2 permutation on the given state.
    fn permute_state_initial(&self, state: &mut [Bn254; WIDTH]) {
        external_initial_permute_state(
            state,
            self.get_initial_constants(),
            add_rc_and_sbox_generic,
            &HLMDSMat4,
        );
    }

    /// Perform the terminal external layers of the Poseidon2 permutation on the given state.
    fn permute_state_terminal(&self, state: &mut [Bn254; WIDTH]) {
        external_terminal_permute_state(
            state,
            self.get_terminal_constants(),
            add_rc_and_sbox_generic,
            &HLMDSMat4,
        );
    }
}

