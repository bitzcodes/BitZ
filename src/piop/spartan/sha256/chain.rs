//! Chained SHA-256 compressions: the Merkle–Damgård chain
//! `H_{i+1} = Compress(H_i, M_i)` for `i < N`, proved as ONE relation whose
//! intermediate chaining values are witness.
//!
//! The batch reuses the independent-compression circuit and its 184 linear
//! constraints per instance, but instance `i` no longer commits its 256
//! chaining-state bits: its state rows of the derived assignment `h_i` are
//! read straight from instance `i - 1`'s committed output cells through a
//! cross-instance F₂ map ([`ChainedPackedSourceMap`]), instance 0's from
//! the public initial state (constants), and the last instance additionally
//! exposes its output in 256 *terminal* rows. The public statement is the
//! block sequence plus the final digest; it is bound by the same uniform
//! public-I/O batching as the independent batch (block slots at every
//! instance, terminal slots equal to the digest at the last instance and to
//! zero elsewhere, where they are structurally zero). Everything else — the
//! runtime prime, the local-row collapse `β = Cᵀeq(·, ξ)`, the rank-one
//! product opening and the virtual BitZ opening — is the independent
//! protocol unchanged; only the map differs.

use crate::piop::spartan::protocol::ProtocolError;
use crate::piop::spartan::protocol::linear::LinearProof;

use crate::piop::spartan::SpartanField as _;
use circuit::linear_map::CscMatrix;
use field::RingOps;
use std::{
    array,
    sync::{Arc, OnceLock},
};

use blake3::Hasher;
use circuit::{
    constraints::ConstraintGenerator,
    sha256::{
        COMPRESSION_HINT_BITS, COMPRESSION_INPUT_BITS, INITIAL_STATE, ROUND_CONSTANTS,
        compression_circuit,
    },
    witgen::{PackedWitness, Witgen},
};

use flock_core::pcs::{
    commit::Commitment,
    ligerito::{ProverConfig as LigProverConfig, VerifierConfig as LigVerifierConfig},
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use {
    crate::{
        ligerito::{LOG_PACKING, packed_vars},
        ligerito_flock::{FlockCommitHint, LigeritoStatementConfig, ResolvedLigerito},
        pcs::{IntegerMatrixLayout, mod_q_num_chunks},
        poly::mle::FactoredMultilinearExtension,
        transcript::traits::Transcript,
    },
    circuit::linear_map::binary::{
        ChainedPackedSourceMap, ChainedPackedSourceParts, PreparedVirtualMap, VirtualMap,
    },
};

use super::super::{
    SpartanError, SpartanField, SpartanMatrixError,
    bitz::SpartanBitzField,
    matrix::eq_table,
    profile::{IopSecurityParams, IopSecurityProfile, Lambda100},
    protocol::{
        FieldConfig,
        binding::{hash_code, profile_code},
        linear::{
            LinearBatching, LinearDomains, LinearProveOptions, LinearRelationSpec, OpeningClaim,
            frame, prove_linear, verify_linear,
        },
    },
};
use super::{
    constraints::{
        SHA256_CONSTRAINTS, SHA256_F_BAR_LIVE_BITS, SHA256_H_BAR_LIVE_BITS, Sha256ConstraintError,
        balanced_binary_params, circuit_matrix_nnz, convert_native_integer_rows,
        max_boolean_linear_residual_bound, packed_domain_vars, validate_instance_capacity,
    },
    inner_sumcheck::Sha256InnerBitSource,
    prime::{Sha256PrimeProfile, sample_sha256_mod_q_context, sha256_instance_facts},
    proof::{
        commit_source_rows_with_config, compact_eq_table, field_from_raw, hash_ligerito_config,
        hash_security_params, hash_usize, instance_vars, local_constraint_vars,
        map_fixes_constant_assignment_local, validate_rows, validate_shared_constant,
        validate_source_params, weighted_byte_tables,
    },
    witness::{Sha256WitnessError, empty_packed_rows, set_flat_packed_bit},
};

/// Committed source bits of one chained compression: the 512 block bits
/// followed by the [`COMPRESSION_HINT_BITS`] hint bits. The 256 chaining-state
/// bits of the independent circuit are NOT committed — they are read from
/// the previous instance's output cells.
pub const SHA256_CHAIN_F_INSTANCE_BITS: usize = 512 + COMPRESSION_HINT_BITS;
/// Chained source bits after adding the shared constant-one coordinate.
pub const SHA256_CHAIN_F_BAR_LIVE_BITS: usize = SHA256_CHAIN_F_INSTANCE_BITS + 1;
/// Terminal rows appended to every instance's derived assignment: the last
/// instance's output bits, structurally zero everywhere else.
pub const SHA256_CHAIN_TERMINAL_BITS: usize = 256;
/// Live derived cells of one chained compression, including its leading
/// constant: the independent circuit's assignment plus the terminal rows.
pub const SHA256_CHAIN_H_BAR_LIVE_BITS: usize = SHA256_H_BAR_LIVE_BITS + SHA256_CHAIN_TERMINAL_BITS;
/// Per-compression derived width after removing the shared constant cell.
pub const SHA256_CHAIN_H_INSTANCE_BITS: usize = SHA256_CHAIN_H_BAR_LIVE_BITS - 1;

const BLOCK_WORDS: usize = 16;
const STATE_WORDS: usize = 8;
const WORD_BITS: usize = 32;
/// Public slots per instance: the block words, then the terminal words.
const CHAIN_PUBLIC_WORDS: usize = BLOCK_WORDS + STATE_WORDS;
const CHAIN_PUBLIC_BITS: usize = CHAIN_PUBLIC_WORDS * WORD_BITS;
/// Local derived stride of the product tensor (`2^15 ≥ 20 713`).
const LOCAL_STRIDE: usize = SHA256_CHAIN_H_BAR_LIVE_BITS.next_power_of_two();
const LOCAL_BITS: usize = LOCAL_STRIDE.ilog2() as usize;
/// The independent circuit's source layout: `[1 | block 512 | state 256 |
/// hints]`.
const STATE_F_COLUMN_BASE: usize = 1 + 512;
const STATE_F_COLUMN_END: usize = STATE_F_COLUMN_BASE + 256;
/// Output word `w`, bit `b` of the independent circuit's source is column
/// `6881 + 33w + b` (the low 32 bits of the final 33-bit sums).
const OUTPUT_F_COLUMN_BASE: usize = 6_881;
const OUTPUT_WORD_STRIDE: usize = 33;
const SHARED_CONSTANT_CELL: usize = 0;

const CHAIN_PROTOCOL_DOMAIN: &[u8] = b"bitz/spartan-sha256-chain/product-linear/v1";
const CHAIN_ASSIGNMENT_BINDING_DOMAIN: &[u8] =
    b"bitz/spartan-sha256-chain-assignment/runtime-prime/v1";
const CHAIN_PUBLIC_STATEMENT_DOMAIN: &[u8] = b"bitz/spartan-sha256-chain/public-statement/v1";
const CHAIN_PUBLIC_IO_BATCH_DOMAIN: &[u8] = b"bitz/spartan-sha256-chain/public-io-batch/v1";
const CHAIN_CONSTANT_ONE_BATCH_DOMAIN: &[u8] = b"bitz/spartan-sha256-chain/constant-one-batch/v1";
const CHAIN_LOCAL_ROW_POINT_DOMAIN: &[u8] = b"bitz/spartan-sha256-chain/local-row-point/v1";
const CHAIN_INSTANCE_POINT_DOMAIN: &[u8] = b"bitz/spartan-sha256-chain/instance-point/v1";
const CHAIN_OPENING_CLAIM_DOMAIN: &[u8] = b"bitz/spartan-sha256-chain-opening/v1";
const CHAIN_RELATION_DIGEST_DOMAIN: &[u8] = b"bitz/sha256-chain/local-relation/v2";

/// Committed source column of output word `w`, bit `b` in the CHAINED
/// layout (the independent layout minus its 256 state cells).
const fn chain_output_f_column(word: usize, bit: usize) -> usize {
    OUTPUT_F_COLUMN_BASE - 256 + OUTPUT_WORD_STRIDE * word + bit
}

/// Derived column of public slot `slot`: block bits `[0, 512)` sit at the
/// circuit's block lifts, terminal bits `[512, 768)` at the terminal rows.
const fn chain_public_h_column(slot: usize) -> usize {
    assert!(slot < CHAIN_PUBLIC_BITS);
    if slot < BLOCK_WORDS * WORD_BITS {
        1 + 256 + slot
    } else {
        SHA256_H_BAR_LIVE_BITS + (slot - BLOCK_WORDS * WORD_BITS)
    }
}

/// Committed source column of public slot `slot` (same order as
/// [`chain_public_h_column`]); terminal slots read the instance's own output
/// cells.
const fn chain_public_f_column(slot: usize) -> usize {
    assert!(slot < CHAIN_PUBLIC_BITS);
    if slot < BLOCK_WORDS * WORD_BITS {
        1 + slot
    } else {
        let terminal = slot - BLOCK_WORDS * WORD_BITS;
        chain_output_f_column(terminal / WORD_BITS, terminal % WORD_BITS)
    }
}

/// One SHA-256 compression, natively (FIPS 180-4 §6.2.2), on parsed words.
pub fn sha256_compress(state: [u32; 8], block: [u32; 16]) -> [u32; 8] {
    let mut w = [0u32; 64];
    w[..16].copy_from_slice(&block);
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(ROUND_CONSTANTS[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    let working = [a, b, c, d, e, f, g, h];
    array::from_fn(|i| state[i].wrapping_add(working[i]))
}

/// Public claim of a chain batch: the blocks `M_0, …, M_{N-1}` and the
/// digest `H_N`, asserting `H_N = Compress(⋯Compress(H_0, M_0)⋯, M_{N-1})`
/// for the prepared batch's initial state `H_0`. The intermediate chaining
/// values `H_1, …, H_{N-1}` are witness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sha256ChainStatement {
    /// One parsed 512-bit block per chained compression, in chain order.
    pub blocks: Vec<[u32; 16]>,
    /// The chaining value after the last block.
    pub digest: [u32; 8],
}

impl Sha256ChainStatement {
    /// The `CHAIN_PUBLIC_WORDS` public words of instance `instance`: its
    /// block, then the terminal words (the digest at the last instance,
    /// zero elsewhere).
    fn instance_words(&self, instance: usize) -> impl Iterator<Item = u32> + '_ {
        let terminal = if instance + 1 == self.blocks.len() {
            self.digest
        } else {
            [0; STATE_WORDS]
        };
        self.blocks[instance].into_iter().chain(terminal)
    }
}

/// The q-independent chained local relation: the independent circuit's
/// constraints over the widened assignment, its four local maps, and the
/// public initial state baked into `first`.
#[derive(Debug)]
struct ChainLocalRelation {
    native_matrix: CscMatrix<Box<[i64]>>,
    local: PreparedVirtualMap,
    prev: PreparedVirtualMap,
    first: PreparedVirtualMap,
    last: PreparedVirtualMap,
    initial_state: [u32; 8],
    digest: [u8; 32],
    max_boolean_residual_bound: field::Uint<2>,
}

static STANDARD_CHAIN_RELATION: OnceLock<Arc<ChainLocalRelation>> = OnceLock::new();

fn chain_local_relation(
    initial_state: [u32; 8],
) -> Result<Arc<ChainLocalRelation>, Sha256ConstraintError> {
    if initial_state == INITIAL_STATE {
        return Ok(Arc::clone(STANDARD_CHAIN_RELATION.get_or_init(|| {
            Arc::new(
                build_chain_local_relation(INITIAL_STATE)
                    .expect("the static SHA-256 chain relation is valid"),
            )
        })));
    }
    build_chain_local_relation(initial_state).map(Arc::new)
}

/// Replays the symbolic compression circuit once and splits its Bit map
/// by source column: block and hint columns stay in `local` (shifted past
/// the removed state cells), state columns become `prev` entries on the
/// previous instance's output cells and, for instance 0, the parity of the
/// initial-state bits each row reads (`first`, constant column). The
/// terminal rows are `last`'s identity copies of the output cells.
fn build_chain_local_relation(
    initial_state: [u32; 8],
) -> Result<ChainLocalRelation, Sha256ConstraintError> {
    let mut generator = ConstraintGenerator::new(COMPRESSION_INPUT_BITS);
    let inputs = generator.inputs();
    let _ = compression_circuit(&mut generator, &inputs);
    let generated = generator.into_matrices();
    assert_eq!(generated.m.row_count(), SHA256_H_BAR_LIVE_BITS);
    assert_eq!(generated.m.column_count(), SHA256_F_BAR_LIVE_BITS);
    assert_eq!(generated.c.row_count(), SHA256_CONSTRAINTS);
    assert_eq!(generated.c.column_count(), SHA256_H_BAR_LIVE_BITS);
    assert_eq!(circuit_matrix_nnz(&generated.a), 0);
    assert_eq!(circuit_matrix_nnz(&generated.b), 0);
    assert_eq!(generated.m.row(0).unwrap().indices(), &[0]);

    let rows = SHA256_CHAIN_H_BAR_LIVE_BITS;
    let mut local_rows: Vec<Vec<(usize, bool)>> = vec![Vec::new(); rows];
    let mut prev_rows: Vec<Vec<(usize, bool)>> = vec![Vec::new(); rows];
    let mut first_rows: Vec<Vec<(usize, bool)>> = vec![Vec::new(); rows];
    let mut last_rows: Vec<Vec<(usize, bool)>> = vec![Vec::new(); rows];
    for (h_row, row) in generated.m.rows().enumerate() {
        let mut initial_parity = false;
        for &f_column in row.indices() {
            if f_column < STATE_F_COLUMN_BASE {
                local_rows[h_row].push((f_column, true));
            } else if f_column < STATE_F_COLUMN_END {
                let state_bit = f_column - STATE_F_COLUMN_BASE;
                let (word, bit) = (state_bit / WORD_BITS, state_bit % WORD_BITS);
                prev_rows[h_row].push((chain_output_f_column(word, bit), true));
                initial_parity ^= (initial_state[word] >> bit) & 1 == 1;
            } else {
                local_rows[h_row].push((f_column - 256, true));
            }
        }
        if initial_parity {
            first_rows[h_row].push((SHARED_CONSTANT_CELL, true));
        }
    }
    for terminal in 0..SHA256_CHAIN_TERMINAL_BITS {
        last_rows[SHA256_H_BAR_LIVE_BITS + terminal].push((
            chain_output_f_column(terminal / WORD_BITS, terminal % WORD_BITS),
            true,
        ));
    }
    let prepared =
        |entries: Vec<Vec<(usize, bool)>>| -> Result<PreparedVirtualMap, Sha256ConstraintError> {
            let matrix = CscMatrix::try_from_rows(SHA256_CHAIN_F_BAR_LIVE_BITS, entries)
                .map_err(SpartanMatrixError::from)?;
            Ok(PreparedVirtualMap::new(matrix)?)
        };
    let local = prepared(local_rows)?;
    let prev = prepared(prev_rows)?;
    let first = prepared(first_rows)?;
    let last = prepared(last_rows)?;

    let native_rows = convert_native_integer_rows(&generated.c)?;
    let native_matrix = CscMatrix::try_from_rows(SHA256_CHAIN_H_BAR_LIVE_BITS, native_rows)
        .map_err(SpartanMatrixError::from)?;
    let max_boolean_residual_bound = max_boolean_linear_residual_bound(&native_matrix);
    let digest = chain_relation_digest(
        &native_matrix,
        &[&local, &prev, &first, &last],
        initial_state,
    )?;

    Ok(ChainLocalRelation {
        native_matrix,
        local,
        prev,
        first,
        last,
        initial_state,
        digest,
        max_boolean_residual_bound,
    })
}

fn chain_relation_digest(
    matrix: &CscMatrix<Box<[i64]>>,
    maps: &[&PreparedVirtualMap; 4],
    initial_state: [u32; 8],
) -> Result<[u8; 32], Sha256ConstraintError> {
    let mut hash = Hasher::new();
    hash.update(CHAIN_RELATION_DIGEST_DOMAIN);
    for map in maps {
        hash.update(&map.digest());
    }
    for word in initial_state {
        hash.update(&word.to_le_bytes());
    }
    hash.update(b"C");
    for value in [matrix.row_count(), matrix.column_count(), matrix.nnz()] {
        super::constraints::hash_usize(&mut hash, value)?;
    }
    for column in matrix.columns() {
        super::constraints::hash_usize(&mut hash, column.len())?;
        for (row, coefficient) in column {
            super::constraints::hash_usize(&mut hash, row)?;
            hash.update(&coefficient.to_le_bytes());
        }
    }
    Ok(*hash.finalize().as_bytes())
}

/// A q-independent chain relation prepared for `N = 2^k` chained
/// compressions from one public initial state.
#[derive(Clone, Debug)]
pub struct PreparedSha256ChainBatch {
    local: Arc<ChainLocalRelation>,
    map: ChainedPackedSourceMap,
    f_layout: IntegerMatrixLayout,
    h_layout: IntegerMatrixLayout,
    instances: usize,
    log_instance_capacity: usize,
    security: IopSecurityParams,
    ligerito: Option<crate::ligerito_flock::ResolvedLigerito>,
}

impl PreparedSha256ChainBatch {
    /// Select only this batch's Ligerito opener; the enclosing target is retained.
    pub fn with_ligerito(
        mut self,
        selection: crate::ligerito_flock::LigeritoSelection,
    ) -> Result<Self, Sha256ConstraintError> {
        let resolved = selection
            .resolve(
                packed_vars(&self.f_layout),
                self.security.ligerito_target_bits,
            )
            .map_err(Sha256ConstraintError::LigeritoConfig)?;
        self.security.adopt_ood_round(resolved.ood_bits())?;
        self.ligerito = Some(resolved);
        Ok(self)
    }

    pub fn ligerito_configuration(
        &self,
    ) -> Result<&crate::ligerito_flock::ResolvedLigerito, Sha256ConstraintError> {
        self.ligerito.as_ref().ok_or_else(|| {
            Sha256ConstraintError::LigeritoConfig(
                "test-only preparation has no production Ligerito policy".into(),
            )
        })
    }

    /// Number of chained compressions `N`.
    pub const fn instances(&self) -> usize {
        self.instances
    }

    /// `k = log₂ N`.
    pub const fn log_instance_capacity(&self) -> usize {
        self.log_instance_capacity
    }

    /// BitZ geometry of the committed Bit source rows.
    pub const fn source_params(&self) -> &IntegerMatrixLayout {
        &self.f_layout
    }

    /// BitZ geometry of the proof-only product tensor `D[local, instance]`
    /// the opening runs against (`2^t` rows of low instance bits).
    pub const fn assignment_params(&self) -> &IntegerMatrixLayout {
        &self.h_layout
    }

    /// The chained Bit map binding the product tensor to the source.
    pub const fn map(&self) -> &ChainedPackedSourceMap {
        &self.map
    }

    /// The public initial chaining value `H_0` baked into the relation.
    pub fn initial_state(&self) -> [u32; 8] {
        self.local.initial_state
    }

    /// Canonical digest of the local relation (constraints, the four local
    /// maps, and the initial state).
    pub fn relation_digest(&self) -> &[u8; 32] {
        &self.local.digest
    }

    /// Conservative bound on `Σ|C_i|` per local row for a Bit assignment.
    pub fn max_boolean_residual_bound(&self) -> &field::Uint<2> {
        &self.local.max_boolean_residual_bound
    }

    /// The instantiated security parameters and their accounting.
    pub const fn security(&self) -> &IopSecurityParams {
        &self.security
    }

    /// The exact native constraint matrix over the widened local assignment.
    pub(super) fn native_matrix(&self) -> &CscMatrix<Box<[i64]>> {
        &self.local.native_matrix
    }
}

/// Prepares the chain relation for `2^log_compressions` chained compressions
/// from the standard initial state at the default profile ([`Lambda100`]).
pub fn prepare_sha256_chain_batch(
    log_compressions: usize,
) -> Result<PreparedSha256ChainBatch, Sha256ConstraintError> {
    prepare_sha256_chain_batch_with_profile::<Lambda100>(log_compressions)
}

/// [`prepare_sha256_chain_batch`] under an explicit single-prime profile.
pub fn prepare_sha256_chain_batch_with_profile<P: IopSecurityProfile>(
    log_compressions: usize,
) -> Result<PreparedSha256ChainBatch, Sha256ConstraintError> {
    prepare_sha256_chain_batch_with_profile_and_initial_state::<P>(log_compressions, INITIAL_STATE)
}

/// [`prepare_sha256_chain_batch_with_profile`] from an arbitrary public
/// initial chaining value (a continuation of an earlier chain, say).
pub fn prepare_sha256_chain_batch_with_profile_and_initial_state<P: IopSecurityProfile>(
    log_compressions: usize,
    initial_state: [u32; 8],
) -> Result<PreparedSha256ChainBatch, Sha256ConstraintError> {
    // The chain uses only the local-major product opening, which needs
    // enough instances to fill a 128-bit packed row.
    if log_compressions < LOG_PACKING {
        return Err(Sha256ConstraintError::InvalidBatchExponent);
    }
    let instances = 1usize
        .checked_shl(
            u32::try_from(log_compressions)
                .map_err(|_| Sha256ConstraintError::InvalidBatchExponent)?,
        )
        .ok_or(Sha256ConstraintError::InvalidBatchExponent)?;
    validate_instance_capacity(instances)?;
    let prepared = prepare_chain_instances::<P>(instances, initial_state)?.with_ligerito(
        crate::ligerito_flock::LigeritoSelection::for_target(P::LIGERITO_TARGET_BITS),
    )?;
    Sha256PrimeProfile::from_security(&prepared.security, prepared.log_instance_capacity)?;
    let max_q_bits =
        u128::BITS as usize - prepared.security.projection_max.leading_zeros() as usize;
    let forest_count = mod_q_num_chunks(&prepared.h_layout, max_q_bits);
    if forest_count != 1 {
        return Err(Sha256ConstraintError::UnsupportedOpeningForestCount {
            actual: forest_count,
        });
    }
    let target = prepared.security.ligerito_target_bits;
    if !(64..=128).contains(&target) {
        return Err(Sha256ConstraintError::UnsupportedLigeritoTargetBits { actual: target });
    }
    Ok(prepared)
}

#[cfg(test)]
pub(super) fn prepare_sha256_chain_batch_for_test(
    log_compressions: usize,
) -> Result<PreparedSha256ChainBatch, Sha256ConstraintError> {
    prepare_chain_instances::<Lambda100>(1usize << log_compressions, INITIAL_STATE)
}

fn prepare_chain_instances<P: IopSecurityProfile>(
    instances: usize,
    initial_state: [u32; 8],
) -> Result<PreparedSha256ChainBatch, Sha256ConstraintError> {
    if instances < 2 || !instances.is_power_of_two() {
        return Err(Sha256ConstraintError::InvalidBatchExponent);
    }
    let log_instance_capacity = instances.ilog2() as usize;
    let exponent = u32::try_from(log_instance_capacity)
        .map_err(|_| Sha256ConstraintError::InvalidBatchExponent)?;
    let local = chain_local_relation(initial_state)?;
    let source_vars = packed_domain_vars(instances, SHA256_CHAIN_F_INSTANCE_BITS)?;
    let f_layout = balanced_binary_params(source_vars);
    let assignment_vars = LOCAL_BITS + log_instance_capacity;
    let t = log_instance_capacity.min(13);
    let h_layout = IntegerMatrixLayout {
        row_vars: t,
        col_vars: assignment_vars - t,
        word_bits: 1,
    };
    let map = ChainedPackedSourceMap::new(
        local.local.clone(),
        local.prev.clone(),
        local.first.clone(),
        local.last.clone(),
        instances,
        h_layout.cells(),
        f_layout.cells(),
    )?;
    let mut facts = sha256_instance_facts(exponent);
    facts.opening_t = u32::try_from(t).map_err(|_| Sha256ConstraintError::InvalidBatchExponent)?;
    let security = P::instantiate(&facts)?;
    Ok(PreparedSha256ChainBatch {
        local,
        map,
        f_layout,
        h_layout,
        instances,
        log_instance_capacity,
        security,
        ligerito: None, // Explicit algebra fixtures omit a production PCS policy.
    })
}

/// Derives the Ligerito configuration selected by the batch's profile.
pub fn sha256_chain_configs(
    prepared: &PreparedSha256ChainBatch,
) -> Result<(LigProverConfig, LigVerifierConfig), ProtocolError> {
    validate_source_params(&prepared.f_layout)?;
    let resolved = prepared.ligerito_configuration()?;
    Ok((resolved.prover().clone(), resolved.verifier().clone()))
}

/// A complete chain witness batch: the packed committed source, the
/// proof-only product tensor, and every chaining value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sha256ChainWitnessBatch {
    /// `[1 | f_0 | … | f_{N-1} | 0…]`, `f_i = [block_i | hints_i]`
    /// ([`SHA256_CHAIN_F_INSTANCE_BITS`] cells each), in BitZ's native
    /// row-then-column order.
    source_rows: Vec<Vec<u64>>,
    /// `D[local, instance] = h_instance[local]` over
    /// [`SHA256_CHAIN_H_BAR_LIVE_BITS`] local cells (local stride `2^15`),
    /// never committed: the chained map binds it to `source_rows`.
    product_assignment_rows: Vec<Vec<u64>>,
    /// `H_0, H_1, …, H_N`: the initial state, every intermediate chaining
    /// value, and the digest.
    states: Vec<[u32; 8]>,
    blocks: Vec<[u32; 16]>,
}

impl Sha256ChainWitnessBatch {
    pub(crate) fn source_rows(&self) -> &[Vec<u64>] {
        &self.source_rows
    }

    pub(crate) fn product_assignment_rows(&self) -> &[Vec<u64>] {
        &self.product_assignment_rows
    }

    /// `H_0, …, H_N` (`N + 1` chaining values).
    pub fn states(&self) -> &[[u32; 8]] {
        &self.states
    }

    /// The final chaining value `H_N`.
    pub fn digest(&self) -> [u32; 8] {
        self.states[self.states.len() - 1]
    }

    /// Number of chained compressions.
    pub fn instances(&self) -> usize {
        self.blocks.len()
    }

    /// The public statement this witness satisfies.
    pub fn statement(&self) -> Sha256ChainStatement {
        Sha256ChainStatement {
            blocks: self.blocks.clone(),
            digest: self.digest(),
        }
    }
}

struct ChainShard {
    f: PackedWitness,
    h_bar: PackedWitness,
    output_bits: [bool; 256],
}

/// Runs the chain natively, then synthesizes every compression's witness
/// (in parallel) and packs the chained source and the product tensor.
pub fn generate_sha256_chain_witnesses(
    prepared: &PreparedSha256ChainBatch,
    blocks: &[[u32; 16]],
) -> Result<Sha256ChainWitnessBatch, Sha256WitnessError> {
    let instances = prepared.instances();
    if blocks.len() != instances
        || prepared.f_layout.word_bits != 1
        || prepared.h_layout.word_bits != 1
        || prepared.f_layout.cells()
            != (1 + instances * SHA256_CHAIN_F_INSTANCE_BITS).next_power_of_two()
        || prepared.h_layout.cells() != instances * LOCAL_STRIDE
    {
        return Err(Sha256WitnessError::InvalidGeometry);
    }
    let mut states = Vec::with_capacity(instances + 1);
    states.push(prepared.initial_state());
    for block in blocks {
        let last = states[states.len() - 1];
        states.push(sha256_compress(last, *block));
    }
    let generate =
        |(instance, block): (usize, &[u32; 16])| -> Result<ChainShard, Sha256WitnessError> {
            let input_bits = compression_input_bits(states[instance], *block);
            let mut generator = Witgen::with_inputs_and_capacity(
                &input_bits,
                COMPRESSION_INPUT_BITS + COMPRESSION_HINT_BITS,
            );
            let output_bits = compression_circuit(&mut generator, &input_bits);
            let (f, h_bar) = generator.into_witnesses();
            if f.bit_len() != COMPRESSION_INPUT_BITS + COMPRESSION_HINT_BITS
                || h_bar.bit_len() != SHA256_H_BAR_LIVE_BITS
            {
                return Err(Sha256WitnessError::UnexpectedCircuitShape);
            }
            let output: [u32; 8] = array::from_fn(|word| {
                (0..WORD_BITS).fold(0u32, |value, bit| {
                    value | (u32::from(output_bits[word * WORD_BITS + bit]) << bit)
                })
            });
            if output != states[instance + 1] {
                return Err(Sha256WitnessError::ChainOutputMismatch);
            }
            Ok(ChainShard {
                f,
                h_bar,
                output_bits,
            })
        };
    #[cfg(feature = "parallel")]
    let shards: Vec<ChainShard> = blocks
        .par_iter()
        .enumerate()
        .map(generate)
        .collect::<Result<_, _>>()?;
    #[cfg(not(feature = "parallel"))]
    let shards: Vec<ChainShard> = blocks
        .iter()
        .enumerate()
        .map(generate)
        .collect::<Result<_, _>>()?;

    let source_rows = pack_chain_source_rows(&shards, &prepared.f_layout);
    let product_assignment_rows = pack_chain_product_rows(&shards, &prepared.h_layout);
    Ok(Sha256ChainWitnessBatch {
        source_rows,
        product_assignment_rows,
        states,
        blocks: blocks.to_vec(),
    })
}

fn compression_input_bits(state: [u32; 8], block: [u32; 16]) -> [bool; COMPRESSION_INPUT_BITS] {
    array::from_fn(|index| {
        let word = if index < 512 {
            block[index / WORD_BITS]
        } else {
            state[(index - 512) / WORD_BITS]
        };
        word >> (index % WORD_BITS) & 1 == 1
    })
}

/// Chained source bit `bit` (`0..SHA256_CHAIN_F_INSTANCE_BITS`) of one
/// instance: the block bits, then the hint bits (the state bits are
/// skipped).
#[inline]
fn chain_source_bit(shard: &ChainShard, bit: usize) -> bool {
    if bit < 512 {
        shard.f.bit(bit)
    } else {
        shard.f.bit(bit + 256)
    }
}

/// Packs `[1 | f_0 | … | f_{N-1} | 0…]`.
fn pack_chain_source_rows(shards: &[ChainShard], params: &IntegerMatrixLayout) -> Vec<Vec<u64>> {
    let mut rows = empty_packed_rows(params);
    set_flat_packed_bit(&mut rows, params, 0);
    for (instance, shard) in shards.iter().enumerate() {
        let start = 1 + instance * SHA256_CHAIN_F_INSTANCE_BITS;
        for bit in 0..SHA256_CHAIN_F_INSTANCE_BITS {
            if chain_source_bit(shard, bit) {
                set_flat_packed_bit(&mut rows, params, start + bit);
            }
        }
    }
    rows
}

/// Packs the local-major product tensor `D[local, instance]` (`2^t` rows =
/// the low instance bits; a column = `(local, high instance bits)`), the
/// terminal rows holding the last instance's output bits only.
fn pack_chain_product_rows(shards: &[ChainShard], params: &IntegerMatrixLayout) -> Vec<Vec<u64>> {
    let instances = shards.len();
    let rows = params.rows();
    debug_assert!(instances.is_power_of_two() && rows <= instances);
    debug_assert_eq!(params.cells(), instances * LOCAL_STRIDE);
    let high_instances = instances / rows;
    let last = instances - 1;
    let build_column = |column: usize| {
        let mut words = vec![0u64; rows.div_ceil(u64::BITS as usize)];
        let local = column / high_instances;
        let high_instance = column % high_instances;
        if local >= SHA256_CHAIN_H_BAR_LIVE_BITS {
            return words;
        }
        let first_instance = high_instance * rows;
        for (word_index, word) in words.iter_mut().enumerate() {
            let first_row = word_index * u64::BITS as usize;
            let mut packed = 0u64;
            for bit in 0..u64::BITS as usize {
                let row = first_row + bit;
                if row >= rows {
                    break;
                }
                let instance = first_instance + row;
                let value = if local < SHA256_H_BAR_LIVE_BITS {
                    shards[instance].h_bar.bit(local)
                } else {
                    instance == last && shards[instance].output_bits[local - SHA256_H_BAR_LIVE_BITS]
                };
                if value {
                    packed |= 1u64 << bit;
                }
            }
            *word = packed;
        }
        words
    };
    #[cfg(feature = "parallel")]
    {
        (0..params.cols())
            .into_par_iter()
            .map(build_column)
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..params.cols()).map(build_column).collect()
    }
}

/// Commits the chained source rows under an explicit Ligerito config.
pub fn commit_sha256_chain_witness_with_config(
    prepared: &PreparedSha256ChainBatch,
    witness: &Sha256ChainWitnessBatch,
    pc: &LigProverConfig,
) -> Result<FlockCommitHint, ProtocolError> {
    validate_chain_ligerito_config(prepared, pc)?;
    validate_rows(&prepared.f_layout, witness.source_rows())?;
    validate_shared_constant(witness.source_rows())?;
    commit_source_rows_with_config(&prepared.f_layout, witness.source_rows().to_vec(), pc)
}

/// Commits the chained source rows under the batch's derived config.
pub fn commit_sha256_chain_witness(
    prepared: &PreparedSha256ChainBatch,
    witness: &Sha256ChainWitnessBatch,
) -> Result<FlockCommitHint, ProtocolError> {
    let (pc, _) = sha256_chain_configs(prepared)?;
    commit_sha256_chain_witness_with_config(prepared, witness, &pc)
}

/// Proves a chain batch against its source commitment.
pub fn prove_sha256_chain<T: Transcript + Send>(
    transcript: &mut T,
    prepared: &PreparedSha256ChainBatch,
    statement: &Sha256ChainStatement,
    witness: &Sha256ChainWitnessBatch,
    hint_f: &FlockCommitHint,
) -> Result<LinearProof, ProtocolError> {
    let (pc, _) = sha256_chain_configs(prepared)?;
    prove_sha256_chain_with_config(transcript, prepared, statement, witness, hint_f, &pc)
}

/// [`prove_sha256_chain`] under an explicit Ligerito config.
pub fn prove_sha256_chain_with_config<T: Transcript + Send>(
    transcript: &mut T,
    prepared: &PreparedSha256ChainBatch,
    statement: &Sha256ChainStatement,
    witness: &Sha256ChainWitnessBatch,
    hint_f: &FlockCommitHint,
    pc: &LigProverConfig,
) -> Result<LinearProof, ProtocolError> {
    prove_linear(
        transcript,
        prepared,
        statement,
        witness,
        hint_f,
        pc,
        LinearProveOptions::default(),
    )
}

/// Verifies a chain proof, re-deriving `q` from the commitment-bound
/// transcript.
pub fn verify_sha256_chain<T: Transcript + Send>(
    transcript: &mut T,
    prepared: &PreparedSha256ChainBatch,
    statement: &Sha256ChainStatement,
    commitment_f: &Commitment,
    proof: &LinearProof,
) -> Result<(), ProtocolError> {
    let (_, vc) = sha256_chain_configs(prepared)?;
    verify_sha256_chain_with_config(transcript, prepared, statement, commitment_f, proof, &vc)
}

/// [`verify_sha256_chain`] under an explicit Ligerito config.
pub fn verify_sha256_chain_with_config<T: Transcript + Send>(
    transcript: &mut T,
    prepared: &PreparedSha256ChainBatch,
    statement: &Sha256ChainStatement,
    commitment_f: &Commitment,
    proof: &LinearProof,
    vc: &LigVerifierConfig,
) -> Result<(), ProtocolError> {
    verify_linear(transcript, prepared, statement, commitment_f, proof, vc)
}

/// The transcript domains of the chain relation.
static CHAIN_DOMAINS: LinearDomains = LinearDomains {
    initial_grinding: b"bitz/spartan-sha256-chain/grinding/initial/v1",
    terminal_grinding: b"bitz/spartan-sha256-chain/grinding/public-batch/v1",
    local_point: CHAIN_LOCAL_ROW_POINT_DOMAIN,
    instance_point: CHAIN_INSTANCE_POINT_DOMAIN,
    constant_one: CHAIN_CONSTANT_ONE_BATCH_DOMAIN,
    public_io: CHAIN_PUBLIC_IO_BATCH_DOMAIN,
    claim_tag: b"sha256-chain-opening-claim",
    claim_domain: CHAIN_OPENING_CLAIM_DOMAIN,
    claim_variant: None,
};

/// The chain batch as the shared linear relation: product-only, opened over
/// its local-major product tensor through the chained map.
impl LinearRelationSpec for PreparedSha256ChainBatch {
    type Statement = Sha256ChainStatement;
    type Witness = Sha256ChainWitnessBatch;
    type Map = ChainedPackedSourceMap;
    type Batching = ChainProductBatching;

    fn domains(&self) -> &'static LinearDomains {
        &CHAIN_DOMAINS
    }

    fn security(&self) -> &IopSecurityParams {
        &self.security
    }

    fn ligerito(&self) -> Result<&ResolvedLigerito, ProtocolError> {
        Ok(self.ligerito_configuration()?)
    }

    fn source_layout(&self) -> &IntegerMatrixLayout {
        &self.f_layout
    }

    fn opened_layout(&self) -> &IntegerMatrixLayout {
        &self.h_layout
    }

    fn map(&self) -> &ChainedPackedSourceMap {
        &self.map
    }

    fn product_layout(&self) -> bool {
        true
    }

    fn local_vars(&self) -> usize {
        local_constraint_vars()
    }

    fn instance_vars(&self) -> usize {
        self.log_instance_capacity
    }

    fn public_io_count(&self) -> usize {
        CHAIN_PUBLIC_BITS
    }

    fn validate_statement(&self, statement: &Sha256ChainStatement) -> Result<(), ProtocolError> {
        validate_chain_statement(self, statement)?;
        validate_chain_geometry(self)
    }

    fn validate_opener_config(
        &self,
        config: &dyn LigeritoStatementConfig,
    ) -> Result<(), ProtocolError> {
        validate_chain_ligerito_config(self, config)
    }

    fn validate_witness(
        &self,
        statement: &Sha256ChainStatement,
        witness: &Sha256ChainWitnessBatch,
        hint: &FlockCommitHint,
    ) -> Result<(), ProtocolError> {
        if witness.instances() != self.instances
            || witness.blocks != statement.blocks
            || witness.digest() != statement.digest
            || witness.states[0] != self.initial_state()
        {
            return Err(ProtocolError::ChainStatementMismatch);
        }
        validate_rows(&self.f_layout, witness.source_rows())?;
        validate_shared_constant(witness.source_rows())?;
        validate_rows(&self.h_layout, witness.product_assignment_rows())?;
        if hint.rows() != witness.source_rows() {
            return Err(ProtocolError::InvalidGeometry);
        }
        Ok(())
    }

    fn statement_frames(
        &self,
        statement: &Sha256ChainStatement,
        commitment: &Commitment,
        config: &dyn LigeritoStatementConfig,
    ) -> Result<(Vec<(&'static [u8], Vec<u8>)>, [u8; 32]), ProtocolError> {
        let statement_binding = chain_statement_binding(self, statement)?;
        let assignment_binding =
            chain_assignment_binding(self, commitment, config, &statement_binding)?;
        let mut frames = vec![
            frame(b"protocol", CHAIN_PROTOCOL_DOMAIN),
            frame(b"integer-relation", self.relation_digest()),
            frame(b"chained-boolean-map", self.map.digest()),
        ];
        for (tag, value) in [
            (&b"instance-vars"[..], self.log_instance_capacity),
            (b"instance-count", self.instances),
            (b"assignment-row-vars", self.h_layout.row_vars),
            (b"assignment-column-vars", self.h_layout.col_vars),
            (b"source-row-vars", self.f_layout.row_vars),
            (b"source-column-vars", self.f_layout.col_vars),
        ] {
            frames.push(frame(tag, (value as u64).to_le_bytes()));
        }
        frames.push(frame(b"public-sha256-chain", statement_binding));
        frames.push(frame(b"assignment-oracle", assignment_binding));
        Ok((frames, assignment_binding))
    }

    fn runtime_prime<T: Transcript>(
        &self,
        transcript: &mut T,
    ) -> Result<field::FpCtx<2>, ProtocolError> {
        let profile =
            Sha256PrimeProfile::from_security(&self.security, self.log_instance_capacity)?;
        let context = sample_sha256_mod_q_context(transcript, profile)?;
        Ok(context)
    }

    fn runtime_relation_frames(&self, config: &FieldConfig) -> Vec<(&'static [u8], Vec<u8>)> {
        vec![
            frame(
                b"runtime-field-modulus",
                SpartanBitzField::canonical_modulus_encoding(config),
            ),
            frame(b"projected-linear-relation", self.relation_digest()),
        ]
    }

    fn batching(
        &self,
        statement: &Sha256ChainStatement,
        local_point: &[SpartanBitzField],
        instance_point: &[SpartanBitzField],
        slot_weights: Vec<SpartanBitzField>,
        public_io_batch: SpartanBitzField,
        constant_one: SpartanBitzField,
        reducer: &field::FpCtx<2>,
        config: &FieldConfig,
    ) -> Result<ChainProductBatching, ProtocolError> {
        let local_row_weights = eq_table(local_point, config).map_err(SpartanError::from)?;
        let beta = {
            let _scope = tracing::info_span!("sha256:local_relation_collapse").entered();
            crate::sumcheck::bridge::repeated::collapse_signed_columns(
                self.native_matrix(),
                &local_row_weights,
                reducer,
            )
            .map_err(SpartanError::from)?
        };
        ChainProductBatching::new(
            self,
            statement,
            instance_point,
            beta,
            slot_weights,
            public_io_batch,
            constant_one,
            config,
        )
    }

    fn product_claim<'a>(
        &'a self,
        batching: &'a ChainProductBatching,
        prime: &'a field::FpCtx<2>,
    ) -> Result<OpeningClaim<'a>, ProtocolError> {
        let config = prime;
        let (row_weights, cols, claimed) =
            chain_product_opening_claim(batching, &self.h_layout, config)?;
        Ok(OpeningClaim {
            rows: Box::new(move |row| {
                row_weights
                    .get(row)
                    .map(|weight| u128::from(config.to_integer(&(field_from_raw(*weight, config)))))
            }),
            cols,
            claimed,
        })
    }

    fn inner_claim<'a>(
        &'a self,
        _point: &[SpartanBitzField],
        _coefficient_evaluation: &SpartanBitzField,
        _final_claim: SpartanBitzField,
        _prime: &'a field::FpCtx<2>,
    ) -> Result<OpeningClaim<'a>, ProtocolError> {
        Err(ProtocolError::UnsupportedDischarge)
    }

    fn inner_bits<'a>(
        &'a self,
        _witness: &'a Sha256ChainWitnessBatch,
    ) -> Result<Box<dyn Sha256InnerBitSource + 'a>, ProtocolError> {
        Err(ProtocolError::UnsupportedDischarge)
    }

    fn opened_rows<'a>(
        &self,
        witness: &'a Sha256ChainWitnessBatch,
    ) -> Result<&'a [Vec<u64>], ProtocolError> {
        Ok(witness.product_assignment_rows())
    }
}

/// The chain's random linear combination of constraint rows, shared-one
/// residual, and public cells as one rank-one product coefficient
/// `V[instance, local] = u_instance · d[local]`: `d = β + α₀·1[local = 0]
/// + α_pub Σ_{p : c_p = local} λ_p`, `μ = α₀ U + α_pub Σ_i u_i public_i`.
pub(crate) struct ChainProductBatching {
    instances: usize,
    instance_point: Vec<SpartanBitzField>,
    local_coefficients: Vec<SpartanBitzField>,
    initial_claim: SpartanBitzField,
}

impl ChainProductBatching {
    #[allow(clippy::too_many_arguments)]
    fn new(
        prepared: &PreparedSha256ChainBatch,
        statement: &Sha256ChainStatement,
        instance_point: &[SpartanBitzField],
        beta: Vec<SpartanBitzField>,
        slot_weights: Vec<SpartanBitzField>,
        public_batch_weight: SpartanBitzField,
        constant_weight: SpartanBitzField,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> Result<Self, ProtocolError> {
        validate_chain_statement(prepared, statement)?;
        if instance_point.len() != instance_vars(prepared.instances)?
            || beta.len() != SHA256_CHAIN_H_BAR_LIVE_BITS
            || slot_weights.len() != CHAIN_PUBLIC_BITS
        {
            return Err(ProtocolError::InvalidGeometry);
        }
        let instance_weights = eq_table(instance_point, field_config)
            .map_err(SpartanError::from)?
            .into_iter()
            .take(prepared.instances)
            .collect::<Vec<_>>();
        let mut active_instance_sum = SpartanBitzField::zero_with_cfg(field_config);
        for weight in &instance_weights {
            active_instance_sum = field_config.add(&(active_instance_sum), &(weight));
        }

        let mut local_coefficients = beta;
        local_coefficients[SHARED_CONSTANT_CELL] = field_config.add(
            &(local_coefficients[SHARED_CONSTANT_CELL]),
            &(&constant_weight),
        );
        for (slot, slot_weight) in slot_weights.iter().enumerate() {
            let public_coefficient =
                field_config.mul(&(public_batch_weight.clone()), &(slot_weight));
            let coefficient = local_coefficients
                .get_mut(chain_public_h_column(slot))
                .ok_or(ProtocolError::InvalidGeometry)?;
            *coefficient = field_config.add(coefficient, &public_coefficient);
        }

        let byte_tables = weighted_byte_tables(&slot_weights, field_config);
        let mut initial_claim = field_config.mul(&(constant_weight), &(&active_instance_sum));
        for (instance, instance_weight) in instance_weights.iter().enumerate() {
            let mut statement_value = SpartanBitzField::zero_with_cfg(field_config);
            for (word_slot, word) in statement.instance_words(instance).enumerate() {
                for byte in 0..4 {
                    let value = ((word >> (8 * byte)) & 0xff) as usize;
                    statement_value = field_config.add(
                        &(statement_value),
                        &(&byte_tables[4 * word_slot + byte][value]),
                    );
                }
            }
            let mut coefficient =
                field_config.mul(&(instance_weight.clone()), &(&public_batch_weight));
            coefficient = field_config.mul(&(coefficient), &(&statement_value));
            initial_claim = field_config.add(&(initial_claim), &(&coefficient));
        }

        Ok(Self {
            instances: prepared.instances,
            instance_point: instance_point.to_vec(),
            local_coefficients,
            initial_claim,
        })
    }
}

/// The chain never takes the legacy inner-sumcheck path.
impl LinearBatching for ChainProductBatching {
    fn initial_claim(&self) -> &SpartanBitzField {
        &self.initial_claim
    }

    fn factored_matrix_mle(
        &self,
        _config: &FieldConfig,
    ) -> Result<FactoredMultilinearExtension<'_, SpartanBitzField>, ProtocolError> {
        Err(ProtocolError::UnsupportedDischarge)
    }

    fn evaluate(
        &self,
        _point: &[SpartanBitzField],
        _config: &FieldConfig,
    ) -> Result<SpartanBitzField, ProtocolError> {
        Err(ProtocolError::UnsupportedDischarge)
    }
}

/// The direct rank-one BitZ claim over the local-major product tensor: the
/// low `t` instance bits select BitZ rows (`eq_low`), a column is `(local,
/// high instance bits)` with weight `d[local] · eq_high`.
fn chain_product_opening_claim(
    batching: &ChainProductBatching,
    h_layout: &IntegerMatrixLayout,
    field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
) -> Result<(Vec<u128>, Vec<u128>, u128), ProtocolError> {
    let instance_vars = instance_vars(batching.instances)?;
    if !batching.instances.is_power_of_two()
        || batching.instance_point.len() != instance_vars
        || h_layout.word_bits != 1
        || h_layout.row_vars > instance_vars
        || h_layout.row_vars + h_layout.col_vars != instance_vars + LOCAL_BITS
        || batching.local_coefficients.len() != SHA256_CHAIN_H_BAR_LIVE_BITS
    {
        return Err(ProtocolError::InvalidGeometry);
    }
    let row_weights =
        compact_eq_table(&batching.instance_point[..h_layout.row_vars], field_config)?;
    let high_weights =
        compact_eq_table(&batching.instance_point[h_layout.row_vars..], field_config)?;
    if row_weights.len() != h_layout.rows()
        || row_weights.len() * high_weights.len() != batching.instances
        || h_layout.cols() != LOCAL_STRIDE * high_weights.len()
    {
        return Err(ProtocolError::InvalidGeometry);
    }
    let high_instances = high_weights.len();
    let coefficient_at = |column: usize| {
        let local_column = column / high_instances;
        if local_column >= batching.local_coefficients.len() {
            return 0;
        }
        let high_weight = field_from_raw(high_weights[column % high_instances], field_config);
        u128::from(field_config.to_integer(
            &(field_config.mul(
                &(batching.local_coefficients[local_column].clone()),
                &(&high_weight),
            )),
        ))
    };
    #[cfg(feature = "parallel")]
    let col_weights = (0..h_layout.cols())
        .into_par_iter()
        .map(coefficient_at)
        .collect::<Vec<_>>();
    #[cfg(not(feature = "parallel"))]
    let col_weights = (0..h_layout.cols()).map(coefficient_at).collect::<Vec<_>>();
    Ok((
        row_weights,
        col_weights,
        u128::from(field_config.to_integer(&(batching.initial_claim))),
    ))
}

fn validate_chain_statement(
    prepared: &PreparedSha256ChainBatch,
    statement: &Sha256ChainStatement,
) -> Result<(), ProtocolError> {
    if statement.blocks.len() != prepared.instances {
        return Err(ProtocolError::InvalidPublicStatementLength {
            expected: prepared.instances,
            actual: statement.blocks.len(),
        });
    }
    Ok(())
}

/// Structural checks the soundness argument leans on: the map's shape, the
/// shared constant, and that every public slot is an identity copy of its
/// committed source cell (block slots through `local`, terminal slots
/// through `last` at the last instance only).
fn validate_chain_geometry(prepared: &PreparedSha256ChainBatch) -> Result<(), ProtocolError> {
    let f_layout = &prepared.f_layout;
    let h_layout = &prepared.h_layout;
    validate_source_params(f_layout)?;
    let map = &prepared.map;
    let parts = map.parts();
    let instance_vars = prepared.log_instance_capacity;
    if h_layout.word_bits != 1
        || h_layout.row_vars < LOG_PACKING
        || h_layout.row_vars > instance_vars
        || h_layout.row_vars + h_layout.col_vars != instance_vars + LOCAL_BITS
        || map.rows() != h_layout.cells()
        || map.cols() != f_layout.cells()
        || map.instances() != prepared.instances
        || map.live_rows() != prepared.instances * SHA256_CHAIN_H_BAR_LIVE_BITS
        || map.live_cols() != 1 + prepared.instances * SHA256_CHAIN_F_INSTANCE_BITS
        || [parts.local, parts.prev, parts.first, parts.last]
            .iter()
            .any(|local| {
                local.rows() != SHA256_CHAIN_H_BAR_LIVE_BITS
                    || local.cols() != SHA256_CHAIN_F_BAR_LIVE_BITS
            })
        || !map_fixes_constant_assignment_local(parts.local)
        || !chain_map_fixes_public_statement(&parts)
        || prepared.native_matrix().row_count() != SHA256_CONSTRAINTS
        || prepared.native_matrix().column_count() != SHA256_CHAIN_H_BAR_LIVE_BITS
    {
        return Err(ProtocolError::InvalidGeometry);
    }
    Ok(())
}

/// Every public derived column is read from exactly one committed cell:
/// block slots from the instance's own block cells (`local`, no other map
/// touches those rows), terminal slots from the instance's own output cells
/// through `last` alone (so they are structurally zero before the last
/// instance).
fn chain_map_fixes_public_statement(parts: &ChainedPackedSourceParts<'_>) -> bool {
    let mut sources: Vec<Option<usize>> = vec![None; CHAIN_PUBLIC_BITS];
    let slot_of = |h_column: usize| -> Option<usize> {
        let block_start = chain_public_h_column(0);
        let block_end = block_start + BLOCK_WORDS * WORD_BITS;
        if (block_start..block_end).contains(&h_column) {
            return Some(h_column - block_start);
        }
        let terminal_start = chain_public_h_column(BLOCK_WORDS * WORD_BITS);
        (terminal_start..terminal_start + SHA256_CHAIN_TERMINAL_BITS)
            .contains(&h_column)
            .then(|| BLOCK_WORDS * WORD_BITS + h_column - terminal_start)
    };
    for (map, allowed_block, allowed_terminal) in [
        (parts.local, true, false),
        (parts.prev, false, false),
        (parts.first, false, false),
        (parts.last, false, true),
    ] {
        for (f_column, entries) in map.matrix().columns().enumerate() {
            for &h_column in entries.indices() {
                let Some(slot) = slot_of(h_column) else {
                    continue;
                };
                let allowed = if slot < BLOCK_WORDS * WORD_BITS {
                    allowed_block
                } else {
                    allowed_terminal
                };
                if !allowed || sources[slot].replace(f_column).is_some() {
                    return false;
                }
            }
        }
    }
    sources
        .into_iter()
        .enumerate()
        .all(|(slot, source)| source == Some(chain_public_f_column(slot)))
}

fn validate_chain_ligerito_config(
    prepared: &PreparedSha256ChainBatch,
    actual: &dyn LigeritoStatementConfig,
) -> Result<(), ProtocolError> {
    let (expected, _) = sha256_chain_configs(prepared)?;
    if ligerito_config_digest(&expected)? != ligerito_config_digest(actual)? {
        return Err(ProtocolError::MismatchedLigeritoConfig);
    }
    Ok(())
}

fn ligerito_config_digest(config: &dyn LigeritoStatementConfig) -> Result<[u8; 32], ProtocolError> {
    let mut hash = Hasher::new();
    hash_ligerito_config(&mut hash, config)?;
    Ok(*hash.finalize().as_bytes())
}

fn chain_statement_binding(
    prepared: &PreparedSha256ChainBatch,
    statement: &Sha256ChainStatement,
) -> Result<[u8; 32], ProtocolError> {
    let mut hash = Hasher::new();
    hash.update(CHAIN_PUBLIC_STATEMENT_DOMAIN);
    for word in prepared.initial_state() {
        hash.update(&word.to_le_bytes());
    }
    hash_usize(&mut hash, statement.blocks.len())?;
    for block in &statement.blocks {
        for word in block {
            hash.update(&word.to_le_bytes());
        }
    }
    for word in statement.digest {
        hash.update(&word.to_le_bytes());
    }
    Ok(*hash.finalize().as_bytes())
}

fn chain_assignment_binding(
    prepared: &PreparedSha256ChainBatch,
    commitment: &Commitment,
    config: &dyn LigeritoStatementConfig,
    statement_binding: &[u8; 32],
) -> Result<[u8; 32], ProtocolError> {
    let mut hash = Hasher::new();
    hash.update(CHAIN_ASSIGNMENT_BINDING_DOMAIN);
    hash.update(&commitment.root);
    hash.update(prepared.relation_digest());
    hash.update(&prepared.map.digest());
    hash.update(statement_binding);
    for value in [
        commitment.params.m,
        commitment.params.log_inv_rate,
        commitment.params.log_batch_size,
        prepared.instances,
        prepared.log_instance_capacity,
        prepared.h_layout.row_vars,
        prepared.h_layout.col_vars,
        prepared.h_layout.word_bits,
        prepared.f_layout.row_vars,
        prepared.f_layout.col_vars,
        prepared.f_layout.word_bits,
    ] {
        hash_usize(&mut hash, value)?;
    }
    hash.update(&[profile_code(commitment.params.profile)]);
    hash.update(&[hash_code(commitment.params.merkle_hash)]);
    hash_security_params(&mut hash, &prepared.security)?;
    hash_ligerito_config(&mut hash, config)?;
    Ok(*hash.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Blake3Transcript;

    fn blocks(instances: usize, seed: u32) -> Vec<[u32; 16]> {
        (0..instances)
            .map(|instance| {
                array::from_fn(|word| {
                    (seed ^ instance as u32 ^ (word as u32) << 8)
                        .wrapping_mul(0x9e37_79b9)
                        .rotate_left(word as u32)
                })
            })
            .collect()
    }

    fn packed_bit(rows: &[Vec<u64>], params: &IntegerMatrixLayout, flat: usize) -> bool {
        let column = flat >> params.row_vars;
        let row = flat & (params.rows() - 1);
        rows[column][row / 64] >> (row % 64) & 1 == 1
    }

    #[test]
    fn native_compression_matches_fips_abc() {
        let block = [
            0x61626380, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00000018,
        ];
        assert_eq!(
            sha256_compress(INITIAL_STATE, block),
            [
                0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
                0xf20015ad,
            ]
        );
    }

    #[test]
    fn chain_local_maps_have_the_expected_geometry() {
        let relation = chain_local_relation(INITIAL_STATE).unwrap();
        for map in [
            &relation.local,
            &relation.prev,
            &relation.first,
            &relation.last,
        ] {
            assert_eq!(map.rows(), SHA256_CHAIN_H_BAR_LIVE_BITS);
            assert_eq!(map.cols(), SHA256_CHAIN_F_BAR_LIVE_BITS);
        }
        // The state columns moved entirely into `prev`; nothing else did.
        assert!(relation.prev.matrix().column(0).unwrap().is_empty());
        assert!(relation.prev.nnz() >= 256);
        assert_eq!(relation.local.nnz() + relation.prev.nnz(), {
            let mut generator = ConstraintGenerator::new(COMPRESSION_INPUT_BITS);
            let inputs = generator.inputs();
            let _ = compression_circuit(&mut generator, &inputs);
            generator
                .into_matrices()
                .m
                .rows()
                .map(|row| row.indices().len())
                .sum::<usize>()
        });
        // Instance 0 reads the initial state through the constant column only.
        assert!(
            relation
                .first
                .matrix()
                .columns()
                .skip(1)
                .all(|column| column.is_empty())
        );
        assert!(relation.first.nnz() > 0);
        // The terminal rows are the last instance's output cells.
        assert_eq!(relation.last.nnz(), SHA256_CHAIN_TERMINAL_BITS);
        for terminal in 0..SHA256_CHAIN_TERMINAL_BITS {
            let column = chain_output_f_column(terminal / 32, terminal % 32);
            assert_eq!(
                relation.last.matrix().column(column).unwrap().indices(),
                &[SHA256_H_BAR_LIVE_BITS + terminal]
            );
        }
        assert_eq!(relation.native_matrix.row_count(), SHA256_CONSTRAINTS);
        assert_eq!(
            relation.native_matrix.column_count(),
            SHA256_CHAIN_H_BAR_LIVE_BITS
        );
        assert!(u128::from(relation.max_boolean_residual_bound) < 1u128 << 111);
        // A different initial state changes only `first` and the digest.
        let other = chain_local_relation([1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        assert_eq!(other.local, relation.local);
        assert_eq!(other.prev, relation.prev);
        assert_eq!(other.last, relation.last);
        assert_ne!(other.first, relation.first);
        assert_ne!(other.digest, relation.digest);
    }

    #[test]
    fn chain_witness_matches_the_map_and_the_native_chain() {
        let prepared = prepare_sha256_chain_batch_for_test(2).unwrap();
        let instances = prepared.instances();
        let blocks = blocks(instances, 0x1234);
        let witness = generate_sha256_chain_witnesses(&prepared, &blocks).unwrap();
        let mut expected = INITIAL_STATE;
        for (instance, block) in blocks.iter().enumerate() {
            assert_eq!(witness.states()[instance], expected);
            expected = sha256_compress(expected, *block);
        }
        assert_eq!(witness.digest(), expected);

        let f_layout = prepared.source_params();
        let h_layout = prepared.assignment_params();
        let map = prepared.map();
        let source = witness.source_rows();
        let product = witness.product_assignment_rows();
        assert!(packed_bit(source, f_layout, 0));
        let mut derived = vec![false; h_layout.cells()];
        for column in 0..f_layout.cells() {
            if !packed_bit(source, f_layout, column) {
                continue;
            }
            for row in map.column_rows(column).unwrap() {
                derived[row] ^= true;
            }
        }
        for (flat, expected) in derived.iter().enumerate() {
            assert_eq!(
                packed_bit(product, h_layout, flat),
                *expected,
                "product cell {flat}"
            );
        }
        // The terminal rows carry the digest at the last instance only.
        for terminal in 0..SHA256_CHAIN_TERMINAL_BITS {
            for instance in 0..instances {
                let flat = (SHA256_H_BAR_LIVE_BITS + terminal) * instances + instance;
                let expected = instance + 1 == instances
                    && (witness.digest()[terminal / 32] >> (terminal % 32)) & 1 == 1;
                assert_eq!(packed_bit(product, h_layout, flat), expected);
            }
        }
        // Every instance's state rows equal the previous chaining value.
        for instance in 0..instances {
            for bit in 0..256 {
                let flat = (1 + bit) * instances + instance;
                let expected = (witness.states()[instance][bit / 32] >> (bit % 32)) & 1 == 1;
                assert_eq!(packed_bit(product, h_layout, flat), expected);
            }
        }
    }

    #[test]
    fn chain_roundtrip_and_public_binding() {
        for selection in [
            crate::ligerito_flock::LigeritoSelection::JOHNSON,
            crate::ligerito_flock::LigeritoSelection::MATCHED_UDR,
        ] {
            const LOG_COMPRESSIONS: usize = 7;
            let prepared = prepare_sha256_chain_batch(LOG_COMPRESSIONS)
                .unwrap()
                .with_ligerito(selection)
                .unwrap();
            let blocks = blocks(prepared.instances(), 0xbeef);
            let witness = generate_sha256_chain_witnesses(&prepared, &blocks).unwrap();
            let statement = witness.statement();
            let (pc, vc) = sha256_chain_configs(&prepared).unwrap();
            let hint = commit_sha256_chain_witness_with_config(&prepared, &witness, &pc).unwrap();

            let mut prover_transcript = Blake3Transcript::new();
            let proof = prove_sha256_chain_with_config(
                &mut prover_transcript,
                &prepared,
                &statement,
                &witness,
                &hint,
                &pc,
            )
            .unwrap();
            assert_eq!(proof.bitz().mfs.len(), 1, "one merged forest");

            let mut verifier_transcript = Blake3Transcript::new();
            verify_sha256_chain_with_config(
                &mut verifier_transcript,
                &prepared,
                &statement,
                &hint.commitment,
                &proof,
                &vc,
            )
            .unwrap();

            let reject = |statement: &Sha256ChainStatement, what: &str| {
                let mut transcript = Blake3Transcript::new();
                assert!(
                    verify_sha256_chain_with_config(
                        &mut transcript,
                        &prepared,
                        statement,
                        &hint.commitment,
                        &proof,
                        &vc,
                    )
                    .is_err(),
                    "{what} must be bound"
                );
            };
            let mut wrong_digest = statement.clone();
            wrong_digest.digest[3] ^= 1 << 17;
            reject(&wrong_digest, "the digest");
            let mut wrong_block = statement.clone();
            wrong_block.blocks[5][9] ^= 1;
            reject(&wrong_block, "a middle block");
            let mut swapped = statement.clone();
            swapped.blocks.swap(0, 1);
            reject(&swapped, "the block order");
            let mut short = statement.clone();
            short.blocks.pop();
            let mut transcript = Blake3Transcript::new();
            assert!(matches!(
                verify_sha256_chain_with_config(
                    &mut transcript,
                    &prepared,
                    &short,
                    &hint.commitment,
                    &proof,
                    &vc
                ),
                Err(ProtocolError::InvalidPublicStatementLength {
                    expected: 128,
                    actual: 127
                })
            ));

            // The prover refuses a statement its witness does not satisfy.
            let mut mismatched_transcript = Blake3Transcript::new();
            assert!(matches!(
                prove_sha256_chain_with_config(
                    &mut mismatched_transcript,
                    &prepared,
                    &wrong_digest,
                    &witness,
                    &hint,
                    &pc,
                ),
                Err(ProtocolError::ChainStatementMismatch)
            ));
        }
    }

    #[test]
    fn chain_preparation_rejects_unsupported_shapes() {
        for exponent in [0, 6, 17] {
            assert!(
                prepare_sha256_chain_batch(exponent).is_err(),
                "2^{exponent}"
            );
        }
        assert!(prepare_sha256_chain_batch_for_test(0).is_err());
        for exponent in 7..=16 {
            let prepared = prepare_sha256_chain_batch(exponent).unwrap();
            assert_eq!(prepared.assignment_params().row_vars, exponent.min(13));
            assert_eq!(
                prepared.assignment_params().row_vars + prepared.assignment_params().col_vars,
                15 + exponent
            );
            sha256_chain_configs(&prepared).unwrap();
        }
    }
}
