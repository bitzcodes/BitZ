//! Product-structured linear SHA-256 PIOP followed by the virtual BitZ opening.
//!
//! The generated SHA circuit has empty R1CS `A` and `B` matrices, so every
//! live row is the integer-linear equation `C h = 0`. Batch rows are challenged
//! as the product of an eight-variable local-row equality tensor and an
//! instance equality tensor. This collapses the local matrix once to
//! `β = Cᵀ eq(·, ξ)`, while the physical witness remains gap-free at
//! `1 + instance * (L - 1) + (local_column - 1)`. Shared-one and public-I/O
//! coefficients are folded into the same local vector. For power-of-two
//! batches, a proof-only local-column × instance view of `h` makes the final
//! coefficient rank one across BitZ's row/column split, so BitZ opens the batched
//! residual directly without an assignment-domain sumcheck. The committed
//! source remains the canonical gap-free `f`; the virtual map binds the
//! proof-only view back to that commitment. Non-power-of-two assignment-row
//! batches retain the legacy inner-sumcheck fallback.

use crate::piop::spartan::protocol::ProtocolError;
use crate::piop::spartan::protocol::linear::LinearProof;

use crate::piop::spartan::SpartanField as _;
use crate::poly::mle::FactoredMultilinearExtension;
#[cfg(test)]
use crate::sumcheck::arithmetic::SumcheckLinearReducer;
#[cfg(test)]
use field::Fp;
use field::{RingOps, Uint};
use std::collections::{HashMap, hash_map::Entry};

use blake3::Hasher;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use flock_core::pcs::{
    commit::Commitment,
    ligerito::{ProverConfig as LigProverConfig, VerifierConfig as LigVerifierConfig},
};

use {
    crate::{
        f2map::cell_count,
        ligerito::LOG_PACKING,
        ligerito_flock::{
            FlockCommitHint, LigeritoStatementConfig, ResolvedLigerito, commit_rs_ligerito_rows,
            validate_ligerito_commitment,
        },
        pcs::IntegerMatrixLayout,
        transcript::traits::Transcript,
    },
    circuit::linear_map::binary::{
        PackedRepeatedVirtualMap, PackedSourceOrder, PackedSourceRepeatedVirtualMap, VirtualMap,
    },
};

use super::super::{
    SpartanError, SpartanField,
    bitz::SpartanBitzField,
    matrix::eq_table,
    profile::IopSecurityParams,
    protocol::{
        FieldConfig,
        binding::{hash_code, profile_code},
        linear::{
            LinearBatching, LinearDomains, LinearProveOptions, LinearRelationSpec, OpeningClaim,
            frame, prove_linear, verify_linear,
        },
    },
    sumcheck::SumcheckError,
};

use super::{
    constraints::{
        PreparedSha256CompressionBatch, PreparedSha256LinearRelation, SHA256_CONSTRAINTS,
        SHA256_H_BAR_LIVE_BITS, SHA256_PUBLIC_WORD_BITS, SHA256_PUBLIC_WORDS,
        sha256_public_f_column, sha256_public_h_column,
    },
    inner_sumcheck::Sha256InnerBitSource,
    prime::sample_sha256_mod_q_context,
    witness::{Sha256CompressionStatement, Sha256CompressionWitnessBatch},
};

#[cfg(test)]
use super::{inner_sumcheck::SHA256_INNER_PREFIX_MAX_VARS, prime::Sha256PrimeProfile};

const SHA256_OPENING_CLAIM_DOMAIN: &[u8] = b"bitz/spartan-sha256-opening/v7";
const SHA256_CONSTANT_ONE_BATCH_DOMAIN: &[u8] = b"bitz/spartan-sha256/constant-one-batch/v3";
const SHA256_PUBLIC_STATEMENT_DOMAIN: &[u8] = b"bitz/spartan-sha256/public-statement/v1";
const SHA256_PUBLIC_IO_BATCH_DOMAIN: &[u8] = b"bitz/spartan-sha256/public-io-batch/v3";
const SHA256_LOCAL_ROW_POINT_DOMAIN: &[u8] = b"bitz/spartan-sha256/local-row-point/v1";
const SHA256_INSTANCE_POINT_DOMAIN: &[u8] = b"bitz/spartan-sha256/instance-point/v1";
const SHA256_SHARED_CONSTANT_CELL: usize = 0;
const SHA256_PROTOCOL_DOMAIN: &[u8] = b"bitz/spartan-sha256-compressions/product-linear/v2";
const SHA256_ASSIGNMENT_BINDING_DOMAIN: &[u8] = b"bitz/spartan-sha256-assignment/runtime-prime/v2";

/// One runtime-field element in Montgomery form, without cloning the shared
/// 128-bit modulus configuration into every dense table entry.

/// Default number of low witness-position variables handled by the packed
/// native-bit prefix kernel.  This is a prover-local performance choice and is
/// deliberately absent from the proof and transcript.
pub const SHA256_DEFAULT_INNER_PREFIX_VARS: usize = 2;

/// Derives the BLAKE3/UDR Ligerito configuration selected by the prepared
/// batch's security profile.
pub fn sha256_compression_configs(
    prepared: &PreparedSha256CompressionBatch,
) -> Result<(LigProverConfig, LigVerifierConfig), ProtocolError> {
    let f_layout = prepared.source_params();
    validate_source_params(f_layout)?;
    let resolved = prepared.ligerito_configuration()?;
    Ok((resolved.prover().clone(), resolved.verifier().clone()))
}

/// Commits packed extended-source rows `[1 | f]` under an explicit config.
pub(super) fn commit_source_rows_with_config(
    f_layout: &IntegerMatrixLayout,
    rows: Vec<Vec<u64>>,
    pc: &LigProverConfig,
) -> Result<FlockCommitHint, ProtocolError> {
    validate_source_params(f_layout)?;
    validate_rows(f_layout, &rows)?;
    validate_shared_constant(&rows)?;
    let hint = commit_rs_ligerito_rows(f_layout, rows, pc);
    validate_ligerito_commitment(&hint.commitment, pc).map_err(ProtocolError::Bitz)?;
    Ok(hint)
}

/// Commits the q-independent source row retained by a packed SHA witness.
pub fn commit_sha256_compression_witness_with_config(
    prepared: &PreparedSha256CompressionBatch,
    witness: &Sha256CompressionWitnessBatch,
    pc: &LigProverConfig,
) -> Result<FlockCommitHint, ProtocolError> {
    validate_ligerito_config_for(prepared, pc)?;
    validate_rows(prepared.source_params(), witness.source_rows())?;
    validate_shared_constant(witness.source_rows())?;
    commit_source_rows_with_config(prepared.source_params(), witness.source_rows().to_vec(), pc)
}

/// Commits a q-independent witness with the prepared security profile.
pub fn commit_sha256_compression_witness(
    prepared: &PreparedSha256CompressionBatch,
    witness: &Sha256CompressionWitnessBatch,
) -> Result<FlockCommitHint, ProtocolError> {
    let (pc, _) = sha256_compression_configs(prepared)?;
    commit_sha256_compression_witness_with_config(prepared, witness, &pc)
}

/// The transcript domains of the compression relation.
static SHA256_DOMAINS: LinearDomains = LinearDomains {
    initial_grinding: b"bitz/spartan-sha256/grinding/initial/v1",
    terminal_grinding: b"bitz/spartan-sha256/grinding/public-batch/v1",
    local_point: SHA256_LOCAL_ROW_POINT_DOMAIN,
    instance_point: SHA256_INSTANCE_POINT_DOMAIN,
    constant_one: SHA256_CONSTANT_ONE_BATCH_DOMAIN,
    public_io: SHA256_PUBLIC_IO_BATCH_DOMAIN,
    claim_tag: b"sha256-opening-claim",
    claim_domain: SHA256_OPENING_CLAIM_DOMAIN,
    claim_variant: Some(b"direct-product"),
};

/// The compression batch as the shared linear relation, over the grid its
/// opening runs against: the proof-only product tensor (direct opening) or
/// the gap-free assignment (legacy inner sumcheck).
struct Sha256CompressionSpec<'p, M> {
    prepared: &'p PreparedSha256CompressionBatch,
    map: &'p M,
    opened: &'p IntegerMatrixLayout,
    /// The product tensor's order; `None` on the legacy path.
    order: Option<PackedSourceOrder>,
    instance_vars: usize,
}

enum Sha256CompressionRelation<'p> {
    Product(Sha256CompressionSpec<'p, PackedSourceRepeatedVirtualMap>),
    Legacy(Sha256CompressionSpec<'p, PackedRepeatedVirtualMap>),
}

impl<'p> Sha256CompressionRelation<'p> {
    /// Validates the batch's geometry and selects its opening path.
    fn new(prepared: &'p PreparedSha256CompressionBatch) -> Result<Self, ProtocolError> {
        let f_layout = prepared.source_params();
        let h_layout = prepared.assignment_params();
        validate_common_geometry(None, prepared.map(), h_layout, f_layout)?;
        let instance_vars = instance_vars(prepared.instances())?;
        match (prepared.product_map(), prepared.product_assignment_params()) {
            (Some(product_map), Some(product_p_h)) => {
                validate_product_geometry(product_map, product_p_h, f_layout)?;
                Ok(Self::Product(Sha256CompressionSpec {
                    prepared,
                    map: product_map,
                    opened: product_p_h,
                    order: Some(product_map.order()),
                    instance_vars,
                }))
            }
            (None, None) => Ok(Self::Legacy(Sha256CompressionSpec {
                prepared,
                map: prepared.map(),
                opened: h_layout,
                order: None,
                instance_vars,
            })),
            _ => Err(ProtocolError::InvalidGeometry),
        }
    }
}

impl<M: VirtualMap> LinearRelationSpec for Sha256CompressionSpec<'_, M> {
    type Statement = [Sha256CompressionStatement];
    type Witness = Sha256CompressionWitnessBatch;
    type Map = M;
    type Batching = ProductLinearBatching;

    fn domains(&self) -> &'static LinearDomains {
        &SHA256_DOMAINS
    }

    fn security(&self) -> &IopSecurityParams {
        self.prepared.security()
    }

    fn ligerito(&self) -> Result<&ResolvedLigerito, ProtocolError> {
        Ok(self.prepared.ligerito_configuration()?)
    }

    fn source_layout(&self) -> &IntegerMatrixLayout {
        self.prepared.source_params()
    }

    fn opened_layout(&self) -> &IntegerMatrixLayout {
        self.opened
    }

    fn map(&self) -> &M {
        self.map
    }

    fn product_layout(&self) -> bool {
        self.order.is_some()
    }

    fn local_vars(&self) -> usize {
        local_constraint_vars()
    }

    fn instance_vars(&self) -> usize {
        self.instance_vars
    }

    fn public_io_count(&self) -> usize {
        SHA256_PUBLIC_WORDS * SHA256_PUBLIC_WORD_BITS
    }

    fn validate_statement(
        &self,
        statement: &[Sha256CompressionStatement],
    ) -> Result<(), ProtocolError> {
        validate_public_statement(self.prepared.instances(), statement)
    }

    fn validate_opener_config(
        &self,
        config: &dyn LigeritoStatementConfig,
    ) -> Result<(), ProtocolError> {
        validate_ligerito_config_for(self.prepared, config)
    }

    fn validate_witness(
        &self,
        _statement: &[Sha256CompressionStatement],
        witness: &Sha256CompressionWitnessBatch,
        hint: &FlockCommitHint,
    ) -> Result<(), ProtocolError> {
        let prepared = self.prepared;
        validate_rows(prepared.source_params(), witness.source_rows())?;
        validate_shared_constant(witness.source_rows())?;
        validate_rows(prepared.assignment_params(), witness.assignment_rows())?;
        match (self.order, witness.product_assignment_rows()) {
            (Some(_), Some(product_rows)) => validate_rows(self.opened, product_rows)?,
            (None, None) => {}
            _ => return Err(ProtocolError::InvalidGeometry),
        }
        if witness.instances() != prepared.instances()
            || witness.outputs().len() != prepared.instances()
            || hint.rows() != witness.source_rows()
        {
            return Err(ProtocolError::InvalidGeometry);
        }
        Ok(())
    }

    fn statement_frames(
        &self,
        statement: &[Sha256CompressionStatement],
        commitment: &Commitment,
        config: &dyn LigeritoStatementConfig,
    ) -> Result<(Vec<(&'static [u8], Vec<u8>)>, [u8; 32]), ProtocolError> {
        let prepared = self.prepared;
        let public_statement_binding = public_statement_binding(statement)?;
        let assignment_binding =
            assignment_binding(prepared, commitment, config, &public_statement_binding)?;
        let mut frames = vec![
            frame(b"protocol", SHA256_PROTOCOL_DOMAIN),
            frame(b"integer-relation", prepared.integer_relation_digest()),
            frame(b"boolean-map", prepared.map().digest()),
        ];
        if let (Some(product_map), Some(product_params)) =
            (prepared.product_map(), prepared.product_assignment_params())
        {
            frames.push(frame(b"product-boolean-map", product_map.digest()));
            frames.push(frame(
                b"product-assignment-row-vars",
                (product_params.row_vars as u64).to_le_bytes(),
            ));
            frames.push(frame(
                b"product-assignment-column-vars",
                (product_params.col_vars as u64).to_le_bytes(),
            ));
        } else {
            frames.push(frame(b"product-boolean-map", b"legacy-inner-sumcheck"));
        }
        for (tag, value) in [
            (&b"instance-vars"[..], prepared.log_instance_capacity()),
            (b"instance-count", prepared.instances()),
            (
                b"assignment-row-vars",
                prepared.assignment_params().row_vars,
            ),
            (
                b"assignment-column-vars",
                prepared.assignment_params().col_vars,
            ),
            (b"source-row-vars", prepared.source_params().row_vars),
            (b"source-column-vars", prepared.source_params().col_vars),
        ] {
            frames.push(frame(tag, (value as u64).to_le_bytes()));
        }
        frames.push(frame(b"public-sha256-io", public_statement_binding));
        frames.push(frame(b"assignment-oracle", assignment_binding));
        Ok((frames, assignment_binding))
    }

    fn runtime_prime<T: Transcript>(
        &self,
        transcript: &mut T,
    ) -> Result<field::FpCtx<2>, ProtocolError> {
        let context = sample_sha256_mod_q_context(transcript, self.prepared.prime_profile())?;
        Ok(context)
    }

    fn runtime_relation_frames(&self, config: &FieldConfig) -> Vec<(&'static [u8], Vec<u8>)> {
        // The exact signed relation was bound before prime sampling.
        // Projection is deterministic from that digest and the runtime
        // modulus, so no dense or padded matrix serialization is needed.
        vec![
            frame(
                b"runtime-field-modulus",
                SpartanBitzField::canonical_modulus_encoding(config),
            ),
            frame(
                b"projected-linear-relation",
                self.prepared.integer_relation_digest(),
            ),
        ]
    }

    fn batching(
        &self,
        statement: &[Sha256CompressionStatement],
        local_point: &[SpartanBitzField],
        instance_point: &[SpartanBitzField],
        slot_weights: Vec<SpartanBitzField>,
        public_io_batch: SpartanBitzField,
        constant_one: SpartanBitzField,
        reducer: &field::FpCtx<2>,
        config: &FieldConfig,
    ) -> Result<ProductLinearBatching, ProtocolError> {
        let local_row_weights = eq_table(local_point, config).map_err(SpartanError::from)?;
        let beta = {
            let _scope = tracing::info_span!("sha256:local_relation_collapse").entered();
            crate::sumcheck::bridge::repeated::collapse_signed_columns(
                self.prepared.linear_relation().native_matrix(),
                &local_row_weights,
                reducer,
            )
            .map_err(SpartanError::from)?
        };
        ProductLinearBatching::new(
            self.prepared,
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
        batching: &'a ProductLinearBatching,
        prime: &'a field::FpCtx<2>,
    ) -> Result<OpeningClaim<'a>, ProtocolError> {
        let order = self.order.ok_or(ProtocolError::InvalidGeometry)?;
        let config = prime;
        let (row_weights, cols, claimed) =
            product_opening_claim(batching, self.opened, order, config)?;
        Ok(OpeningClaim {
            rows: Box::new(move |row| row_weights.canonical_weight(row, config)),
            cols,
            claimed,
        })
    }

    fn inner_claim<'a>(
        &'a self,
        point: &[SpartanBitzField],
        coefficient_evaluation: &SpartanBitzField,
        final_claim: SpartanBitzField,
        prime: &'a field::FpCtx<2>,
    ) -> Result<OpeningClaim<'a>, ProtocolError> {
        let config = prime;
        let equality = FactoredEqualityWeights::new(point, self.opened.row_vars, config)?;
        let (row_weights, cols, claimed) = linear_opening_claim(
            self.prepared,
            self.opened,
            equality,
            coefficient_evaluation,
            final_claim,
            &config,
        )?;
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

    fn inner_bits<'a>(
        &'a self,
        witness: &'a Sha256CompressionWitnessBatch,
    ) -> Result<Box<dyn Sha256InnerBitSource + 'a>, ProtocolError> {
        if self.order.is_some() {
            return Err(ProtocolError::InvalidGeometry);
        }
        let layout = self.opened;
        Ok(Box::new(move |flat_column| {
            packed_flat_bit(witness.assignment_rows(), layout, flat_column)
        }))
    }

    fn opened_rows<'a>(
        &self,
        witness: &'a Sha256CompressionWitnessBatch,
    ) -> Result<&'a [Vec<u64>], ProtocolError> {
        match self.order {
            Some(_) => witness
                .product_assignment_rows()
                .ok_or(ProtocolError::InvalidGeometry),
            None => Ok(witness.assignment_rows()),
        }
    }
}

/// Proves with an explicit packed-prefix width and Ligerito configuration.
///
/// `prefix_vars` must be in `0..=4`. It only selects the prover kernel for the
/// non-power-of-two legacy fallback; direct product openings ignore it. It is
/// neither serialized nor absorbed.
#[allow(clippy::too_many_arguments)]
pub fn prove_sha256_compressions_with_prefix_vars_and_config<T: Transcript + Send>(
    transcript: &mut T,
    prepared: &PreparedSha256CompressionBatch,
    public_statement: &[Sha256CompressionStatement],
    witness: &Sha256CompressionWitnessBatch,
    hint_f: &FlockCommitHint,
    prefix_vars: usize,
    pc: &LigProverConfig,
) -> Result<LinearProof, ProtocolError> {
    let options = LinearProveOptions { prefix_vars };
    match Sha256CompressionRelation::new(prepared)? {
        Sha256CompressionRelation::Product(spec) => prove_linear(
            transcript,
            &spec,
            public_statement,
            witness,
            hint_f,
            pc,
            options,
        ),
        Sha256CompressionRelation::Legacy(spec) => prove_linear(
            transcript,
            &spec,
            public_statement,
            witness,
            hint_f,
            pc,
            options,
        ),
    }
}

/// Proves with the default packed-prefix kernel and an explicit Ligerito
/// configuration.
#[allow(clippy::too_many_arguments)]
pub fn prove_sha256_compressions_with_config<T: Transcript + Send>(
    transcript: &mut T,
    prepared: &PreparedSha256CompressionBatch,
    public_statement: &[Sha256CompressionStatement],
    witness: &Sha256CompressionWitnessBatch,
    hint_f: &FlockCommitHint,
    pc: &LigProverConfig,
) -> Result<LinearProof, ProtocolError> {
    prove_sha256_compressions_with_prefix_vars_and_config(
        transcript,
        prepared,
        public_statement,
        witness,
        hint_f,
        SHA256_DEFAULT_INNER_PREFIX_VARS,
        pc,
    )
}

/// Proves with the prepared batch's derived Ligerito configuration.
pub fn prove_sha256_compressions<T: Transcript + Send>(
    transcript: &mut T,
    prepared: &PreparedSha256CompressionBatch,
    public_statement: &[Sha256CompressionStatement],
    witness: &Sha256CompressionWitnessBatch,
    hint_f: &FlockCommitHint,
) -> Result<LinearProof, ProtocolError> {
    let (pc, _) = sha256_compression_configs(prepared)?;
    prove_sha256_compressions_with_config(
        transcript,
        prepared,
        public_statement,
        witness,
        hint_f,
        &pc,
    )
}

/// Proves with an explicit prover-local packed-prefix width and the prepared
/// batch's derived Ligerito configuration.
pub fn prove_sha256_compressions_with_prefix_vars<T: Transcript + Send>(
    transcript: &mut T,
    prepared: &PreparedSha256CompressionBatch,
    public_statement: &[Sha256CompressionStatement],
    witness: &Sha256CompressionWitnessBatch,
    hint_f: &FlockCommitHint,
    prefix_vars: usize,
) -> Result<LinearProof, ProtocolError> {
    let (pc, _) = sha256_compression_configs(prepared)?;
    prove_sha256_compressions_with_prefix_vars_and_config(
        transcript,
        prepared,
        public_statement,
        witness,
        hint_f,
        prefix_vars,
        &pc,
    )
}

/// Verifies the runtime-prime SHA profile while independently re-deriving `q`
/// from the commitment-bound transcript.
#[allow(clippy::too_many_arguments)]
pub fn verify_sha256_compressions_with_config<T: Transcript + Send>(
    transcript: &mut T,
    prepared: &PreparedSha256CompressionBatch,
    public_statement: &[Sha256CompressionStatement],
    commitment_f: &Commitment,
    proof: &LinearProof,
    vc: &LigVerifierConfig,
) -> Result<(), ProtocolError> {
    match Sha256CompressionRelation::new(prepared)? {
        Sha256CompressionRelation::Product(spec) => {
            verify_linear(transcript, &spec, public_statement, commitment_f, proof, vc)
        }
        Sha256CompressionRelation::Legacy(spec) => {
            verify_linear(transcript, &spec, public_statement, commitment_f, proof, vc)
        }
    }
}

/// Verifies with the prepared batch's derived Ligerito configuration.
pub fn verify_sha256_compressions<T: Transcript + Send>(
    transcript: &mut T,
    prepared: &PreparedSha256CompressionBatch,
    public_statement: &[Sha256CompressionStatement],
    commitment_f: &Commitment,
    proof: &LinearProof,
) -> Result<(), ProtocolError> {
    let (_, vc) = sha256_compression_configs(prepared)?;
    verify_sha256_compressions_with_config(
        transcript,
        prepared,
        public_statement,
        commitment_f,
        proof,
        &vc,
    )
}

/// Random linear combination of the SHA rows, shared-one residual, and public
/// SHA cells folded into one product-structured oracle.
///
/// Let `v_r = eq(r, ξ)`, `u_i = eq(i, η)`, `U = Σ_{i<N} u_i`, and
/// `β = Cᵀv`. The local vector `d` adds the shared-one and public-I/O
/// coefficients to `β`. Its gap-free physical realization is
///
/// - `V[0] = U d[0]`, and
/// - `V[1 + i(L - 1) + (c - 1)] = u_i d[c]` for `c > 0`.
///
/// Thus the native matrix is collapsed once, independently of the number of
/// instances. The opening claim `μ` is the same product-structured batching
/// of the public values: `α₀ U + α_pub Σ_i u_i public_i`.
struct ProductLinearBatching {
    instances: usize,
    instance_point: Vec<SpartanBitzField>,
    instance_weights: Vec<SpartanBitzField>,
    local_coefficients: Vec<SpartanBitzField>,
    shared_coefficient: SpartanBitzField,
    initial_claim: SpartanBitzField,
}

impl ProductLinearBatching {
    #[allow(clippy::too_many_arguments)]
    fn new(
        prepared: &PreparedSha256CompressionBatch,
        public_statement: &[Sha256CompressionStatement],
        instance_point: &[SpartanBitzField],
        beta: Vec<SpartanBitzField>,
        slot_weights: Vec<SpartanBitzField>,
        public_batch_weight: SpartanBitzField,
        constant_weight: SpartanBitzField,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> Result<Self, ProtocolError> {
        validate_public_statement(prepared.instances(), public_statement)?;
        if instance_point.len() != instance_vars(prepared.instances())?
            || beta.len() != SHA256_H_BAR_LIVE_BITS
            || slot_weights.len() != SHA256_PUBLIC_WORDS * SHA256_PUBLIC_WORD_BITS
        {
            return Err(ProtocolError::InvalidGeometry);
        }

        let instance_weights = eq_table(instance_point, field_config)
            .map_err(SpartanError::from)?
            .into_iter()
            .take(prepared.instances())
            .collect::<Vec<_>>();
        let mut active_instance_sum = SpartanBitzField::zero_with_cfg(field_config);
        for weight in &instance_weights {
            active_instance_sum = field_config.add(&(active_instance_sum), &(weight));
        }

        let mut local_coefficients = beta;
        local_coefficients[SHA256_SHARED_CONSTANT_CELL] = field_config.add(
            &(local_coefficients[SHA256_SHARED_CONSTANT_CELL]),
            &(&constant_weight),
        );
        for (slot, slot_weight) in slot_weights.iter().enumerate() {
            let word_slot = slot / SHA256_PUBLIC_WORD_BITS;
            let bit = slot % SHA256_PUBLIC_WORD_BITS;
            let local_column = sha256_public_h_column(word_slot, bit);
            let public_coefficient =
                field_config.mul(&(public_batch_weight.clone()), &(slot_weight));
            let coefficient = local_coefficients
                .get_mut(local_column)
                .ok_or(ProtocolError::InvalidGeometry)?;
            *coefficient = field_config.add(coefficient, &public_coefficient);
        }
        let shared_coefficient = field_config.mul(
            &(active_instance_sum.clone()),
            &(&local_coefficients[SHA256_SHARED_CONSTANT_CELL]),
        );

        // Compress each eight-bit public dot product into a lookup table once.
        // This computes μ without visiting all 1,024 public bits per instance.
        let byte_tables = weighted_byte_tables(&slot_weights, field_config);
        let mut initial_claim = field_config.mul(&(constant_weight), &(&active_instance_sum));
        for (instance, statement) in public_statement.iter().enumerate() {
            let mut statement_value = SpartanBitzField::zero_with_cfg(field_config);
            for (word_slot, word) in statement.words().enumerate() {
                for byte in 0..4 {
                    let value = ((word >> (8 * byte)) & 0xff) as usize;
                    statement_value = field_config.add(
                        &(statement_value),
                        &(&byte_tables[4 * word_slot + byte][value]),
                    );
                }
            }
            let mut coefficient = field_config.mul(
                &(instance_weights[instance].clone()),
                &(&public_batch_weight),
            );
            coefficient = field_config.mul(&(coefficient), &(&statement_value));
            initial_claim = field_config.add(&(initial_claim), &(&coefficient));
        }

        Ok(Self {
            instances: prepared.instances(),
            instance_point: instance_point.to_vec(),
            instance_weights,
            local_coefficients,
            shared_coefficient,
            initial_claim,
        })
    }

    const fn initial_claim(&self) -> &SpartanBitzField {
        &self.initial_claim
    }

    fn factored_matrix_mle(
        &self,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> Result<FactoredMultilinearExtension<'_, SpartanBitzField>, ProtocolError> {
        FactoredMultilinearExtension::with_leading_value(
            (1 + self.instance_weights.len() * (self.local_coefficients.len() - 1))
                .next_power_of_two()
                .ilog2() as usize,
            self.shared_coefficient.clone(),
            &self.instance_weights,
            &self.local_coefficients[1..],
            field_config,
        )
        .map_err(|_| {
            SpartanError::from(
                crate::piop::spartan::sumcheck::SumcheckError::InvalidProductDimensions,
            )
        })
        .map_err(ProtocolError::from)
    }

    #[cfg(test)]
    fn coefficient(
        &self,
        flat_column: usize,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> Result<SpartanBitzField, SumcheckError> {
        if flat_column == SHA256_SHARED_CONSTANT_CELL {
            return Ok(self.shared_coefficient.clone());
        }
        let Some(offset) = flat_column.checked_sub(1) else {
            return Ok(SpartanBitzField::zero_with_cfg(field_config));
        };
        let instance = offset / super::constraints::SHA256_H_INSTANCE_BITS;
        if instance >= self.instances {
            return Ok(SpartanBitzField::zero_with_cfg(field_config));
        }
        let local_column = 1 + offset % super::constraints::SHA256_H_INSTANCE_BITS;
        let coefficient = self
            .local_coefficients
            .get(local_column)
            .ok_or(SumcheckError::InvalidProductDimensions)?;
        Ok(field_config.mul(&(self.instance_weights[instance].clone()), &(coefficient)))
    }

    fn evaluate(
        &self,
        assignment_point: &[SpartanBitzField],
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> Result<SpartanBitzField, ProtocolError> {
        let constant_equality = equality_at_zero(assignment_point, field_config)?;
        let zero = SpartanBitzField::zero_with_cfg(field_config);
        let repeated_nonconstant = evaluate_affine_equality_repetition(
            self.instances,
            0,
            super::constraints::SHA256_H_INSTANCE_BITS,
            &[],
            assignment_point,
            Some(&self.instance_point),
            self.local_coefficients
                .iter()
                .enumerate()
                .skip(1)
                .filter(|(_, coefficient)| **coefficient != zero)
                .map(|(column, coefficient)| (0, column, coefficient.clone())),
            field_config,
        )?;
        Ok(field_config.add(
            &(field_config.mul(&(self.shared_coefficient.clone()), &(&constant_equality))),
            &(&repeated_nonconstant),
        ))
    }

    #[cfg(test)]
    fn evaluate_dense(
        &self,
        assignment_equality: &FactoredEqualityWeights,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> Result<SpartanBitzField, ProtocolError> {
        let mut evaluation = field_config.mul(
            &(self.shared_coefficient.clone()),
            &(&assignment_equality
                .evaluate(SHA256_SHARED_CONSTANT_CELL, field_config)
                .ok_or(ProtocolError::InvalidGeometry)?),
        );
        for instance in 0..self.instances {
            for local_column in 1..self.local_coefficients.len() {
                let flat_column =
                    1 + instance * super::constraints::SHA256_H_INSTANCE_BITS + local_column - 1;
                let equality = assignment_equality
                    .evaluate(flat_column, field_config)
                    .ok_or(ProtocolError::InvalidGeometry)?;
                let coefficient = field_config.mul(
                    &(self.instance_weights[instance].clone()),
                    &(&self.local_coefficients[local_column]),
                );
                evaluation = field_config.add(
                    &(evaluation),
                    &(&(field_config.mul(&(coefficient), &(&equality)))),
                );
            }
        }
        Ok(evaluation)
    }
}

impl LinearBatching for ProductLinearBatching {
    fn initial_claim(&self) -> &SpartanBitzField {
        Self::initial_claim(self)
    }

    fn factored_matrix_mle(
        &self,
        config: &FieldConfig,
    ) -> Result<FactoredMultilinearExtension<'_, SpartanBitzField>, ProtocolError> {
        Self::factored_matrix_mle(self, config)
    }

    fn evaluate(
        &self,
        point: &[SpartanBitzField],
        config: &FieldConfig,
    ) -> Result<SpartanBitzField, ProtocolError> {
        Self::evaluate(self, point, config)
    }
}

/// Legacy public-only batching retained for dense equivalence tests.
#[cfg(test)]
struct PublicLinearBatching {
    instances: usize,
    constant_weight: SpartanBitzField,
    public_batch_weight: SpartanBitzField,
    instance_point: Vec<SpartanBitzField>,
    scaled_instance_weights: Vec<SpartanBitzField>,
    slot_weights: Vec<SpartanBitzField>,
    initial_claim: SpartanBitzField,
}

#[cfg(test)]
impl PublicLinearBatching {
    #[allow(clippy::too_many_arguments)]
    fn new(
        prepared: &PreparedSha256CompressionBatch,
        public_statement: &[Sha256CompressionStatement],
        public_instance_point: &[SpartanBitzField],
        slot_weights: Vec<SpartanBitzField>,
        public_batch_weight: SpartanBitzField,
        constant_weight: SpartanBitzField,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> Result<Self, ProtocolError> {
        validate_public_statement(prepared.instances(), public_statement)?;
        if public_instance_point.len() != instance_vars(prepared.instances())?
            || slot_weights.len() != SHA256_PUBLIC_WORDS * SHA256_PUBLIC_WORD_BITS
        {
            return Err(ProtocolError::InvalidGeometry);
        }
        let instance_weights =
            eq_table(public_instance_point, field_config).map_err(SpartanError::from)?;
        let scaled_instance_weights = instance_weights
            .into_iter()
            .take(prepared.instances())
            .map(|weight| field_config.mul(&(weight), &(&public_batch_weight)))
            .collect::<Vec<_>>();

        // A statement has 1,024 public bits.  Compress each eight-bit dot
        // product into a 256-entry lookup table once, then evaluate each of
        // the 32 public words with four lookups.  This preserves the exact
        // field-valued bit dot product while replacing 1,024 per-instance
        // bit tests and up to 1,024 field multiplications with 128 lookups,
        // additions, and one multiplication.
        let byte_tables = weighted_byte_tables(&slot_weights, field_config);
        let mut initial_claim = constant_weight.clone();
        for (instance, statement) in public_statement.iter().enumerate() {
            let mut statement_value = SpartanBitzField::zero_with_cfg(field_config);
            for (word_slot, word) in statement.words().enumerate() {
                for byte in 0..4 {
                    let value = ((word >> (8 * byte)) & 0xff) as usize;
                    statement_value = field_config.add(
                        &(statement_value),
                        &(&byte_tables[4 * word_slot + byte][value]),
                    );
                }
            }
            initial_claim = field_config.add(
                &(initial_claim),
                &(&(field_config.mul(
                    &(scaled_instance_weights[instance].clone()),
                    &(&statement_value),
                ))),
            );
        }

        Ok(Self {
            instances: prepared.instances(),
            constant_weight,
            public_batch_weight,
            instance_point: public_instance_point.to_vec(),
            scaled_instance_weights,
            slot_weights,
            initial_claim,
        })
    }

    const fn initial_claim(&self) -> &SpartanBitzField {
        &self.initial_claim
    }

    fn coefficient(
        &self,
        flat_column: usize,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> Result<SpartanBitzField, SumcheckError> {
        if flat_column == SHA256_SHARED_CONSTANT_CELL {
            return Ok(self.constant_weight.clone());
        }
        let Some(packed_offset) = flat_column.checked_sub(1) else {
            return Ok(SpartanBitzField::zero_with_cfg(field_config));
        };
        let instance = packed_offset / super::constraints::SHA256_H_INSTANCE_BITS;
        if instance >= self.instances {
            return Ok(SpartanBitzField::zero_with_cfg(field_config));
        }
        let local_column = 1 + packed_offset % super::constraints::SHA256_H_INSTANCE_BITS;
        let Some(slot) = public_bit_index_for_h_column(local_column) else {
            return Ok(SpartanBitzField::zero_with_cfg(field_config));
        };
        Ok(field_config.mul(
            &(self.scaled_instance_weights[instance].clone()),
            &(&self.slot_weights[slot]),
        ))
    }

    fn evaluate(
        &self,
        assignment_point: &[SpartanBitzField],
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> Result<SpartanBitzField, ProtocolError> {
        let constant_equality = equality_at_zero(assignment_point, field_config)?;
        let repeated_public = evaluate_affine_equality_repetition(
            self.instances,
            0,
            super::constraints::SHA256_H_INSTANCE_BITS,
            &[],
            assignment_point,
            Some(&self.instance_point),
            self.slot_weights
                .iter()
                .enumerate()
                .map(|(slot, coefficient)| {
                    let word_slot = slot / SHA256_PUBLIC_WORD_BITS;
                    let bit = slot % SHA256_PUBLIC_WORD_BITS;
                    (
                        0,
                        sha256_public_h_column(word_slot, bit),
                        coefficient.clone(),
                    )
                }),
            field_config,
        )?;
        let mut evaluation =
            field_config.mul(&(self.constant_weight.clone()), &(&constant_equality));
        evaluation = field_config.add(
            &(evaluation),
            &(&(field_config.mul(&(self.public_batch_weight.clone()), &(&repeated_public)))),
        );
        Ok(evaluation)
    }

    #[cfg(test)]
    fn evaluate_dense(
        &self,
        assignment_equality: &FactoredEqualityWeights,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> Result<SpartanBitzField, ProtocolError> {
        let mut evaluation = field_config.mul(
            &(self.constant_weight.clone()),
            &(&assignment_equality
                .evaluate(SHA256_SHARED_CONSTANT_CELL, field_config)
                .ok_or(ProtocolError::InvalidGeometry)?),
        );
        for instance in 0..self.instances {
            for word_slot in 0..SHA256_PUBLIC_WORDS {
                for bit in 0..SHA256_PUBLIC_WORD_BITS {
                    let slot = word_slot * SHA256_PUBLIC_WORD_BITS + bit;
                    let local_column = sha256_public_h_column(word_slot, bit);
                    let flat_column =
                        1 + instance * super::constraints::SHA256_H_INSTANCE_BITS + local_column
                            - 1;
                    let equality = assignment_equality
                        .evaluate(flat_column, field_config)
                        .ok_or(ProtocolError::InvalidGeometry)?;
                    let coefficient = field_config.mul(
                        &(self.scaled_instance_weights[instance].clone()),
                        &(&self.slot_weights[slot]),
                    );
                    evaluation = field_config.add(
                        &(evaluation),
                        &(&(field_config.mul(&(coefficient), &(&equality)))),
                    );
                }
            }
        }
        Ok(evaluation)
    }
}

pub(super) fn weighted_byte_tables(
    weights: &[SpartanBitzField],
    field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
) -> Vec<Vec<SpartanBitzField>> {
    debug_assert_eq!(weights.len() % 8, 0);
    weights
        .chunks_exact(8)
        .map(|byte_weights| {
            let mut table = Vec::with_capacity(256);
            table.push(SpartanBitzField::zero_with_cfg(field_config));
            for value in 1usize..256 {
                let bit = value.trailing_zeros() as usize;
                let previous = value & (value - 1);
                table.push(field_config.add(&(table[previous].clone()), &(&byte_weights[bit])));
            }
            table
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn linear_opening_claim(
    prepared: &PreparedSha256CompressionBatch,
    h_layout: &IntegerMatrixLayout,
    mut assignment_equality: FactoredEqualityWeights,
    collapsed_evaluation: &SpartanBitzField,
    inner_claim: SpartanBitzField,
    field_config: &FieldConfig,
) -> Result<(Vec<u128>, Vec<u128>, u128), ProtocolError> {
    if assignment_equality.len() != h_layout.cells()
        || assignment_equality.low_vars != h_layout.row_vars
        || assignment_equality.low.len() != h_layout.rows()
        || assignment_equality.high.len() != h_layout.cols()
        || prepared.linear_assignment_column_count() > h_layout.cells()
    {
        return Err(ProtocolError::InvalidGeometry);
    }
    assignment_equality.scale(collapsed_evaluation, field_config);
    let row_weights = assignment_equality.low;
    let col_weights = assignment_equality
        .high
        .iter()
        .map(|weight| u128::from(field_config.to_integer(&(field_from_raw(*weight, field_config)))))
        .collect::<Vec<_>>();
    Ok((
        row_weights,
        col_weights,
        u128::from(field_config.to_integer(&(inner_claim))),
    ))
}

enum ProductRowWeights {
    InstanceOnly(Vec<u128>),
    LocalAndInstance {
        local: Vec<u128>,
        low_instance: Vec<u128>,
        local_domain: usize,
    },
}

impl ProductRowWeights {
    fn canonical_weight(
        &self,
        row: usize,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> Option<u128> {
        match self {
            Self::InstanceOnly(weights) => weights.get(row).map(|weight| {
                u128::from(field_config.to_integer(&(field_from_raw(*weight, field_config))))
            }),
            Self::LocalAndInstance {
                local,
                low_instance,
                local_domain,
            } => {
                let local_column = row & (local_domain - 1);
                let instance = row / local_domain;
                let local_weight = field_from_raw(*local.get(local_column)?, field_config);
                let instance_weight = field_from_raw(*low_instance.get(instance)?, field_config);
                Some(u128::from(field_config.to_integer(
                    &(field_config.mul(&(local_weight), &(&instance_weight))),
                )))
            }
        }
    }
}

/// Builds the direct rank-one BitZ claim without materializing `u ⊗ d`.
fn product_opening_claim(
    batching: &ProductLinearBatching,
    h_layout: &IntegerMatrixLayout,
    layout: PackedSourceOrder,
    field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
) -> Result<(ProductRowWeights, Vec<u128>, u128), ProtocolError> {
    let instance_vars = instance_vars(batching.instances)?;
    if !batching.instances.is_power_of_two()
        || batching.instance_point.len() != instance_vars
        || h_layout.word_bits != 1
        || h_layout.row_vars + h_layout.col_vars != instance_vars + 15
        || batching.local_coefficients.len() != SHA256_H_BAR_LIVE_BITS
    {
        return Err(ProtocolError::InvalidGeometry);
    }

    let (row_weights, col_weights) = match layout {
        PackedSourceOrder::LocalMajor => {
            if h_layout.row_vars > instance_vars {
                return Err(ProtocolError::InvalidGeometry);
            }
            let low =
                compact_eq_table(&batching.instance_point[..h_layout.row_vars], field_config)?;
            let high =
                compact_eq_table(&batching.instance_point[h_layout.row_vars..], field_config)?;
            if low.len() != h_layout.rows()
                || low.len() * high.len() != batching.instances
                || h_layout.cols() != SHA256_H_BAR_LIVE_BITS.next_power_of_two() * high.len()
            {
                return Err(ProtocolError::InvalidGeometry);
            }

            let high_instances = high.len();
            let coefficient_at = |column: usize| {
                let local_column = column / high_instances;
                if local_column >= batching.local_coefficients.len() {
                    return 0;
                }
                let high_instance = column % high_instances;
                let high_weight = field_from_raw(high[high_instance], field_config);
                u128::from(field_config.to_integer(
                    &(field_config.mul(
                        &(batching.local_coefficients[local_column].clone()),
                        &(&high_weight),
                    )),
                ))
            };
            #[cfg(feature = "parallel")]
            let columns = (0..h_layout.cols())
                .into_par_iter()
                .map(coefficient_at)
                .collect::<Vec<_>>();
            #[cfg(not(feature = "parallel"))]
            let columns = (0..h_layout.cols()).map(coefficient_at).collect::<Vec<_>>();
            (ProductRowWeights::InstanceOnly(low), columns)
        }
        PackedSourceOrder::InstanceMajor => {
            let local_domain = SHA256_H_BAR_LIVE_BITS.next_power_of_two();
            let local_vars = local_domain.ilog2() as usize;
            if h_layout.row_vars < local_vars || h_layout.row_vars > local_vars + instance_vars {
                return Err(ProtocolError::InvalidGeometry);
            }
            let low_instance_vars = h_layout.row_vars - local_vars;
            let low_instance =
                compact_eq_table(&batching.instance_point[..low_instance_vars], field_config)?;
            let high =
                compact_eq_table(&batching.instance_point[low_instance_vars..], field_config)?;
            if local_domain * low_instance.len() != h_layout.rows()
                || low_instance.len() * high.len() != batching.instances
                || high.len() != h_layout.cols()
            {
                return Err(ProtocolError::InvalidGeometry);
            }
            let local = batching
                .local_coefficients
                .iter()
                .map(raw_montgomery)
                .chain(std::iter::repeat_n(
                    raw_montgomery(&SpartanBitzField::zero_with_cfg(field_config)),
                    local_domain - batching.local_coefficients.len(),
                ))
                .collect();
            let columns = high
                .into_iter()
                .map(|weight| {
                    u128::from(field_config.to_integer(&(field_from_raw(weight, field_config))))
                })
                .collect();
            (
                ProductRowWeights::LocalAndInstance {
                    local,
                    low_instance,
                    local_domain,
                },
                columns,
            )
        }
    };

    Ok((
        row_weights,
        col_weights,
        u128::from(field_config.to_integer(&(batching.initial_claim))),
    ))
}

#[cfg(test)]
fn flat_constraint_vars(prepared: &PreparedSha256CompressionBatch) -> Result<usize, ProtocolError> {
    let live_rows = prepared.linear_row_count();
    let domain = live_rows
        .checked_next_power_of_two()
        .ok_or(ProtocolError::InvalidGeometry)?;
    Ok(domain.ilog2() as usize)
}

pub(super) const fn local_constraint_vars() -> usize {
    SHA256_CONSTRAINTS.next_power_of_two().ilog2() as usize
}

pub(super) fn instance_vars(instances: usize) -> Result<usize, ProtocolError> {
    let domain = instances
        .checked_next_power_of_two()
        .ok_or(ProtocolError::InvalidGeometry)?;
    Ok(domain.ilog2() as usize)
}

#[cfg(test)]
fn public_instance_point(
    constraint_point: &[SpartanBitzField],
    instances: usize,
) -> Result<&[SpartanBitzField], ProtocolError> {
    let vars = instance_vars(instances)?;
    let start = constraint_point
        .len()
        .checked_sub(vars)
        .ok_or(ProtocolError::InvalidGeometry)?;
    constraint_point
        .get(start..)
        .ok_or(ProtocolError::InvalidGeometry)
}

#[cfg(test)]
fn collapse_flat_linear_column(
    prepared: &PreparedSha256CompressionBatch,
    relation: &PreparedSha256LinearRelation,
    constraint_weights: &[u128],
    reducer: &field::FpCtx<2>,
    flat_column: usize,
) -> Result<SpartanBitzField, SumcheckError> {
    let zero = SpartanBitzField::zero_with_cfg(relation.config());
    let live_columns = prepared.linear_assignment_column_count();
    if flat_column >= prepared.assignment_params().cells() {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    if flat_column >= live_columns {
        return Ok(zero);
    }

    let (instance_start, instance_end, local_column) = if flat_column == 0 {
        (0, prepared.instances(), 0)
    } else {
        let offset = flat_column - 1;
        let instance = offset / super::constraints::SHA256_H_INSTANCE_BITS;
        let local_column = 1 + offset % super::constraints::SHA256_H_INSTANCE_BITS;
        (instance, instance + 1, local_column)
    };
    let column = prepared
        .linear_relation()
        .native_matrix()
        .column(local_column)
        .ok_or(SumcheckError::InvalidProductDimensions)?;
    let mut accumulator = <field::FpCtx<2> as SumcheckLinearReducer>::accumulator_zero(reducer);
    for instance in instance_start..instance_end {
        for (local_row, coefficient) in column {
            let flat_row = instance
                .checked_mul(SHA256_CONSTRAINTS)
                .and_then(|base| base.checked_add(local_row))
                .ok_or(SumcheckError::InvalidProductDimensions)?;
            let weight = constraint_weights
                .get(flat_row)
                .ok_or(SumcheckError::InvalidProductDimensions)?;
            let weight = field_from_raw(*weight, relation.config());
            let negative_weight;
            let selected_weight = if *coefficient < 0 {
                negative_weight = relation.config().sub(&(zero.clone()), &(&weight));
                &negative_weight
            } else {
                &weight
            };
            <field::FpCtx<2> as SumcheckLinearReducer>::multiply_accumulate(
                reducer,
                &mut accumulator,
                selected_weight,
                &coefficient.unsigned_abs(),
            );
        }
    }
    <field::FpCtx<2> as SumcheckLinearReducer>::reduce(reducer, accumulator, relation.config())
}

/// Evaluates the repeated flat relation without expanding its `instances`
/// copies.  For every local nonzero `(r, c)`, the repeated coordinates are
///
/// `row(i) = 184 i + r`, `column(i) = 20,456 i + c`.
///
/// Reading `i` from least-significant bit to most-significant bit turns both
/// affine indices into bounded binary carry recurrences. Terms that reach the
/// same pair of carries are algebraically identical for all remaining bits and
/// are merged. If `S` is the maximum carry frontier (independent of the number
/// of instances for this fixed local SHA relation), the contraction costs
/// `O((nnz(C) + S) * log(domain))`, rather than materializing or visiting
/// `instances * nnz(C)` repeated entries.
#[cfg(test)]
fn evaluate_repeated_flat_linear_collapse(
    prepared: &PreparedSha256CompressionBatch,
    relation: &PreparedSha256LinearRelation,
    constraint_point: &[SpartanBitzField],
    assignment_point: &[SpartanBitzField],
) -> Result<SpartanBitzField, ProtocolError> {
    if constraint_point.len() != flat_constraint_vars(prepared)?
        || assignment_point.len()
            != prepared.assignment_params().row_vars + prepared.assignment_params().col_vars
    {
        return Err(ProtocolError::InvalidGeometry);
    }

    let constant_column = relation
        .matrix()
        .column(SHA256_SHARED_CONSTANT_CELL)
        .ok_or(ProtocolError::InvalidGeometry)?;
    let constant_evaluation = evaluate_affine_equality_repetition(
        prepared.instances(),
        SHA256_CONSTRAINTS,
        0,
        constraint_point,
        assignment_point,
        None,
        constant_column
            .into_iter()
            .map(|(row, coefficient)| (row, 0, coefficient.clone())),
        relation.config(),
    )?;

    let nonconstant_terms =
        relation
            .matrix()
            .columns()
            .enumerate()
            .skip(1)
            .flat_map(|(column, entries)| {
                entries
                    .into_iter()
                    .map(move |(row, coefficient)| (row, column, coefficient.clone()))
            });
    let nonconstant_evaluation = evaluate_affine_equality_repetition(
        prepared.instances(),
        SHA256_CONSTRAINTS,
        super::constraints::SHA256_H_INSTANCE_BITS,
        constraint_point,
        assignment_point,
        None,
        nonconstant_terms,
        relation.config(),
    )?;

    Ok(relation
        .config()
        .add(&(constant_evaluation), &(&nonconstant_evaluation)))
}

/// Contracts a weighted set of affine-offset pairs against two equality
/// tensors, optionally also weighting the repetition index by a third
/// equality tensor.  The latter is used for public-I/O batching.
#[allow(clippy::too_many_arguments)]
fn evaluate_affine_equality_repetition<I>(
    instances: usize,
    left_stride: usize,
    right_stride: usize,
    left_point: &[SpartanBitzField],
    right_point: &[SpartanBitzField],
    instance_point: Option<&[SpartanBitzField]>,
    terms: I,
    field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
) -> Result<SpartanBitzField, ProtocolError>
where
    I: IntoIterator<Item = (usize, usize, SpartanBitzField)>,
{
    if instances == 0 {
        return Err(ProtocolError::InvalidGeometry);
    }
    let input_vars = instance_vars(instances)?;
    if instance_point.is_some_and(|point| point.len() != input_vars) {
        return Err(ProtocolError::InvalidGeometry);
    }
    let rounds = left_point.len().max(right_point.len());
    if input_vars > rounds {
        return Err(ProtocolError::InvalidGeometry);
    }
    let input_capacity = 1usize
        .checked_shl(u32::try_from(input_vars).map_err(|_| ProtocolError::InvalidGeometry)?)
        .ok_or(ProtocolError::InvalidGeometry)?;
    let bounded = instances != input_capacity;

    // Each round has eight possible `(left_bit, right_bit, instance_bit)`
    // products.  Precomputing them removes two field multiplications from
    // every carry-state transition.  `None` marks a one bit beyond a tensor's
    // declared domain and therefore an overflowing affine index.
    let one = SpartanBitzField::one_with_cfg(field_config);
    let bit_factors = |point: &[SpartanBitzField], round: usize| {
        point.get(round).map_or_else(
            || [Some(one.clone()), None],
            |challenge| {
                [
                    Some(field_config.sub(&(one.clone()), &(challenge))),
                    Some(challenge.clone()),
                ]
            },
        )
    };
    let mut round_weights: Vec<[Option<SpartanBitzField>; 8]> = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let left = bit_factors(left_point, round);
        let right = bit_factors(right_point, round);
        let input = instance_point.map_or_else(
            || [Some(one.clone()), Some(one.clone())],
            |point| bit_factors(point, round),
        );
        round_weights.push(std::array::from_fn(|index| {
            let mut value = left[index & 1].clone()?;
            value = field_config.mul(&(value), right[(index >> 1) & 1].as_ref()?);
            value = field_config.mul(&(value), input[(index >> 2) & 1].as_ref()?);
            Some(value)
        }));
    }

    // State is `(left carry, right carry, borrow)`.  For a partial batch,
    // `borrow` is the carry of the little-endian subtraction `i - instances`;
    // it is one exactly when the completed instance index satisfies `i < N`.
    let zero = SpartanBitzField::zero_with_cfg(field_config);
    let terms = terms.into_iter();
    let (lower_bound, _) = terms.size_hint();
    let mut states = HashMap::with_capacity(lower_bound);
    for (left_offset, right_offset, coefficient) in terms {
        match states.entry((left_offset, right_offset, false)) {
            Entry::Vacant(entry) => {
                if coefficient != zero {
                    entry.insert(coefficient);
                }
            }
            Entry::Occupied(mut entry) => {
                *entry.get_mut() = field_config.add(entry.get(), &coefficient);
                if *entry.get() == zero {
                    entry.remove();
                }
            }
        }
    }

    for (round, weights) in round_weights.iter().enumerate() {
        let input_active = round < input_vars;
        let instance_bit = if input_active {
            (instances >> round) & 1
        } else {
            0
        };
        let mut next = HashMap::with_capacity(states.len().saturating_mul(2));
        for ((left_carry, right_carry, borrow), value) in states {
            for bit in 0..=usize::from(input_active) {
                let left = left_carry
                    .checked_add(left_stride * bit)
                    .ok_or(ProtocolError::InvalidGeometry)?;
                let right = right_carry
                    .checked_add(right_stride * bit)
                    .ok_or(ProtocolError::InvalidGeometry)?;
                let left_bit = left & 1;
                let right_bit = right & 1;
                let weight_index = left_bit | (right_bit << 1) | (bit << 2);
                let Some(weight) = weights[weight_index].as_ref() else {
                    continue;
                };
                let next_borrow = bounded && bit < instance_bit + usize::from(borrow);
                let term = field_config.mul(&(value.clone()), &(weight));
                if term == zero {
                    continue;
                }
                match next.entry((left >> 1, right >> 1, next_borrow)) {
                    Entry::Vacant(entry) => {
                        entry.insert(term);
                    }
                    Entry::Occupied(mut entry) => {
                        *entry.get_mut() = field_config.add(entry.get(), &term);
                        if *entry.get() == zero {
                            entry.remove();
                        }
                    }
                }
            }
        }
        states = next;
    }

    let mut evaluation = zero;
    for ((left_carry, right_carry, borrow), value) in states {
        if left_carry == 0 && right_carry == 0 && (!bounded || borrow) {
            evaluation = field_config.add(&(evaluation), &(&value));
        }
    }
    Ok(evaluation)
}

fn equality_at_zero(
    point: &[SpartanBitzField],
    field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
) -> Result<SpartanBitzField, ProtocolError> {
    let one = SpartanBitzField::one_with_cfg(field_config);
    Ok(point.iter().fold(one.clone(), |product, challenge| {
        field_config.mul(
            &(product),
            &(&(field_config.sub(&(one.clone()), &(challenge)))),
        )
    }))
}

#[cfg(test)]
fn evaluate_flat_linear_collapse_dense(
    prepared: &PreparedSha256CompressionBatch,
    relation: &PreparedSha256LinearRelation,
    constraint_weights: &[u128],
    assignment_weights: &FactoredEqualityWeights,
) -> Result<SpartanBitzField, ProtocolError> {
    if assignment_weights.len() != prepared.assignment_params().cells()
        || constraint_weights.len()
            != 1usize
                .checked_shl(
                    u32::try_from(flat_constraint_vars(prepared)?)
                        .map_err(|_| ProtocolError::InvalidGeometry)?,
                )
                .ok_or(ProtocolError::InvalidGeometry)?
    {
        return Err(ProtocolError::InvalidGeometry);
    }

    let zero = SpartanBitzField::zero_with_cfg(relation.config());
    let mut evaluation = zero.clone();
    for (local_column, column) in relation.matrix().columns().enumerate() {
        if column.is_empty() {
            continue;
        }
        for instance in 0..prepared.instances() {
            let flat_column = prepared
                .flat_assignment_column(instance, local_column)
                .ok_or(ProtocolError::InvalidGeometry)?;
            let mut column_value = zero.clone();
            for (local_row, coefficient) in column {
                let flat_row = prepared
                    .flat_constraint_row(instance, local_row)
                    .ok_or(ProtocolError::InvalidGeometry)?;
                let constraint_weight =
                    field_from_raw(constraint_weights[flat_row], relation.config());
                column_value = relation.config().add(
                    &column_value,
                    &relation.config().mul(&constraint_weight, coefficient),
                );
            }
            let assignment_weight = assignment_weights
                .evaluate(flat_column, relation.config())
                .ok_or(ProtocolError::InvalidGeometry)?;
            evaluation = relation.config().add(
                &evaluation,
                &relation.config().mul(&column_value, &assignment_weight),
            );
        }
    }
    Ok(evaluation)
}

/// Tensor-factorized little-endian equality table. The low `t` coordinates
/// select BitZ's packed row and the high `s` coordinates select its column, so
/// the two vectors can be passed to the integer opening without constructing
/// the full `2^(t+s)` equality table.
struct FactoredEqualityWeights {
    low: Vec<u128>,
    high: Vec<u128>,
    low_vars: usize,
    len: usize,
}

impl FactoredEqualityWeights {
    fn new(
        point: &[SpartanBitzField],
        low_vars: usize,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> Result<Self, ProtocolError> {
        if low_vars > point.len() {
            return Err(ProtocolError::InvalidGeometry);
        }
        let low = compact_eq_table(&point[..low_vars], field_config)?;
        let high = compact_eq_table(&point[low_vars..], field_config)?;
        let vars = u32::try_from(point.len()).map_err(|_| ProtocolError::InvalidGeometry)?;
        let len = 1usize
            .checked_shl(vars)
            .ok_or(ProtocolError::InvalidGeometry)?;
        Ok(Self {
            low,
            high,
            low_vars,
            len,
        })
    }

    const fn len(&self) -> usize {
        self.len
    }

    fn scale(
        &mut self,
        factor: &SpartanBitzField,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) {
        let scale = |weight: &mut u128| {
            let value = field_config.mul(&(field_from_raw(*weight, field_config)), &(factor));
            *weight = raw_montgomery(&value);
        };
        #[cfg(feature = "parallel")]
        if self.high.len() >= 1 << 13 && rayon::current_num_threads() > 1 {
            self.high.par_iter_mut().for_each(scale);
        } else {
            self.high.iter_mut().for_each(scale);
        }
        #[cfg(not(feature = "parallel"))]
        self.high.iter_mut().for_each(scale);
    }

    #[cfg(test)]
    fn evaluate(
        &self,
        index: usize,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> Option<SpartanBitzField> {
        if index >= self.len {
            return None;
        }
        let low_mask = self.low.len() - 1;
        let low = field_from_raw(*self.low.get(index & low_mask)?, field_config);
        let high = field_from_raw(*self.high.get(index >> self.low_vars)?, field_config);
        Some(field_config.mul(&(low), &(&high)))
    }
}

/// Builds `eq(boolean_index, point)` in little-endian index order while
/// storing only each element's two Montgomery limbs. A full `Fp`
/// carries its runtime modulus configuration, which would otherwise multiply
/// the memory of the flat SHA domains by roughly five.
pub(super) fn compact_eq_table(
    point: &[SpartanBitzField],
    field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
) -> Result<Vec<u128>, ProtocolError> {
    let vars = u32::try_from(point.len()).map_err(|_| ProtocolError::InvalidGeometry)?;
    let table_len = 1usize
        .checked_shl(vars)
        .ok_or(ProtocolError::InvalidGeometry)?;
    let mut table = vec![0; table_len];
    table[0] = raw_montgomery(&SpartanBitzField::one_with_cfg(field_config));

    for (coordinate, challenge) in point.iter().enumerate() {
        let half = 1usize
            .checked_shl(u32::try_from(coordinate).map_err(|_| ProtocolError::InvalidGeometry)?)
            .ok_or(ProtocolError::InvalidGeometry)?;
        let (zero_children, one_children) = table[..2 * half].split_at_mut(half);
        let expand = |(zero_child, one_child): (&mut u128, &mut u128)| {
            let parent = field_from_raw(*zero_child, field_config);
            let high = field_config.mul(&(parent.clone()), &(challenge));
            *zero_child = raw_montgomery(&(field_config.sub(&(parent), &(&high))));
            *one_child = raw_montgomery(&high);
        };
        #[cfg(feature = "parallel")]
        if half >= 1 << 13 && rayon::current_num_threads() > 1 {
            zero_children
                .par_iter_mut()
                .zip(one_children.par_iter_mut())
                .for_each(expand);
        } else {
            zero_children
                .iter_mut()
                .zip(one_children.iter_mut())
                .for_each(expand);
        }
        #[cfg(not(feature = "parallel"))]
        zero_children
            .iter_mut()
            .zip(one_children.iter_mut())
            .for_each(expand);
    }

    Ok(table)
}

#[inline]
pub(super) fn raw_montgomery(value: &SpartanBitzField) -> u128 {
    let words = value.as_montgomery_integer().as_words();
    u128::from(words[0]) | (u128::from(words[1]) << 64)
}

#[inline]
pub(super) fn field_from_raw(
    value: u128,
    field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
) -> SpartanBitzField {
    field_config.from_montgomery_integer(Uint::from(value))
}

pub(super) fn validate_source_params(f_layout: &IntegerMatrixLayout) -> Result<(), ProtocolError> {
    let host_bits = usize::BITS as usize;
    if f_layout.word_bits != 1
        || f_layout.row_vars < LOG_PACKING
        || f_layout.row_vars.saturating_add(f_layout.col_vars) > 126
        || f_layout.row_vars >= host_bits
        || f_layout.col_vars >= host_bits
    {
        return Err(ProtocolError::InvalidGeometry);
    }
    Ok(())
}

fn validate_common_geometry(
    linear_relation: Option<&PreparedSha256LinearRelation>,
    map: &PackedRepeatedVirtualMap,
    h_layout: &IntegerMatrixLayout,
    f_layout: &IntegerMatrixLayout,
) -> Result<(), ProtocolError> {
    validate_source_params(f_layout)?;
    if h_layout.word_bits != 1
        || h_layout.row_vars < LOG_PACKING
        || h_layout.row_vars.saturating_add(h_layout.col_vars) > 126
        || map.rows() != cell_count(h_layout)
        || map.cols() != cell_count(f_layout)
        || map.local().rows() != SHA256_H_BAR_LIVE_BITS
        || map.local().cols() != super::constraints::SHA256_F_BAR_LIVE_BITS
        || !map_fixes_constant_assignment(map)
        || !map_fixes_public_statement(map)
    {
        return Err(ProtocolError::InvalidGeometry);
    }
    if linear_relation.is_some_and(|relation| {
        relation.matrix().row_count() != SHA256_CONSTRAINTS
            || relation.matrix().column_count() != SHA256_H_BAR_LIVE_BITS
    }) {
        return Err(ProtocolError::InvalidGeometry);
    }
    Ok(())
}

fn validate_product_geometry(
    map: &PackedSourceRepeatedVirtualMap,
    h_layout: &IntegerMatrixLayout,
    f_layout: &IntegerMatrixLayout,
) -> Result<(), ProtocolError> {
    validate_source_params(f_layout)?;
    let instance_vars = map.instances().ilog2() as usize;
    let local_stride = SHA256_H_BAR_LIVE_BITS.next_power_of_two();
    let (t_min, t_max, expected_live_rows) = match map.order() {
        PackedSourceOrder::LocalMajor => (
            LOG_PACKING,
            instance_vars,
            map.instances() * SHA256_H_BAR_LIVE_BITS,
        ),
        PackedSourceOrder::InstanceMajor => (
            local_stride.ilog2() as usize,
            local_stride.ilog2() as usize + instance_vars,
            map.instances() * local_stride,
        ),
    };
    if h_layout.word_bits != 1
        || h_layout.row_vars < t_min
        || h_layout.row_vars > t_max
        || h_layout.row_vars.saturating_add(h_layout.col_vars) != instance_vars + 15
        || map.rows() != cell_count(h_layout)
        || map.cols() != cell_count(f_layout)
        || map.live_rows() != expected_live_rows
        || map.local_stride()
            != match map.order() {
                PackedSourceOrder::LocalMajor => map.instances(),
                PackedSourceOrder::InstanceMajor => local_stride,
            }
        || map.local().rows() != SHA256_H_BAR_LIVE_BITS
        || map.local().cols() != super::constraints::SHA256_F_BAR_LIVE_BITS
        || !map_fixes_constant_assignment_local(map.local())
        || !map_fixes_public_statement_local(map.local())
    {
        return Err(ProtocolError::InvalidGeometry);
    }
    Ok(())
}

fn map_fixes_constant_assignment(map: &PackedRepeatedVirtualMap) -> bool {
    map_fixes_constant_assignment_local(map.local())
}

pub(super) fn map_fixes_constant_assignment_local(
    map: &circuit::linear_map::binary::PreparedVirtualMap,
) -> bool {
    let mut constant_source = None;
    for (column, entries) in map.matrix().columns().enumerate() {
        for row in entries.indices() {
            if *row == SHA256_SHARED_CONSTANT_CELL && constant_source.replace(column).is_some() {
                return false;
            }
        }
    }
    constant_source == Some(SHA256_SHARED_CONSTANT_CELL)
}

fn map_fixes_public_statement(map: &PackedRepeatedVirtualMap) -> bool {
    map_fixes_public_statement_local(map.local())
}

fn map_fixes_public_statement_local(map: &circuit::linear_map::binary::PreparedVirtualMap) -> bool {
    const PUBLIC_BITS: usize = SHA256_PUBLIC_WORDS * SHA256_PUBLIC_WORD_BITS;

    let mut sources = [None; PUBLIC_BITS];
    for (f_column, entries) in map.matrix().columns().enumerate() {
        for &h_column in entries.indices() {
            let Some(index) = public_bit_index_for_h_column(h_column) else {
                continue;
            };
            if sources[index].replace(f_column).is_some() {
                return false;
            }
        }
    }

    sources.into_iter().enumerate().all(|(index, source)| {
        let word_slot = index / SHA256_PUBLIC_WORD_BITS;
        let bit = index % SHA256_PUBLIC_WORD_BITS;
        source == Some(sha256_public_f_column(word_slot, bit))
    })
}

fn public_bit_index_for_h_column(h_column: usize) -> Option<usize> {
    const INPUT_WORDS: usize = 24;
    const OUTPUT_WORD_STRIDE: usize = 33;

    let input_start = sha256_public_h_column(0, 0);
    let input_end = sha256_public_h_column(INPUT_WORDS - 1, SHA256_PUBLIC_WORD_BITS - 1) + 1;
    if (input_start..input_end).contains(&h_column) {
        return Some(h_column - input_start);
    }

    let output_start = sha256_public_h_column(INPUT_WORDS, 0);
    let offset = h_column.checked_sub(output_start)?;
    let output_word = offset / OUTPUT_WORD_STRIDE;
    let bit = offset % OUTPUT_WORD_STRIDE;
    if output_word >= SHA256_PUBLIC_WORDS - INPUT_WORDS || bit >= SHA256_PUBLIC_WORD_BITS {
        return None;
    }
    Some((INPUT_WORDS + output_word) * SHA256_PUBLIC_WORD_BITS + bit)
}

pub(super) fn validate_rows(
    p: &IntegerMatrixLayout,
    rows: &[Vec<u64>],
) -> Result<(), ProtocolError> {
    let words = p.rows().div_ceil(64);
    if rows.len() != p.cols() || rows.iter().any(|row| row.len() != words) {
        return Err(ProtocolError::InvalidGeometry);
    }
    Ok(())
}

/// Reads the conceptual flat sequence in BitZ's native packed order. Low `t`
/// index bits select the packed row and high `s` bits select the column, so
/// adjacent SHA instances remain adjacent in the legacy sumcheck oracle and
/// canonical virtual-map indices.
fn packed_flat_bit(
    rows: &[Vec<u64>],
    p: &IntegerMatrixLayout,
    flat_cell: usize,
) -> Result<u64, SumcheckError> {
    if flat_cell >= p.cells() || rows.len() != p.cols() {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    let column = flat_cell >> p.row_vars;
    let row = flat_cell & (p.rows() - 1);
    let words = rows
        .get(column)
        .ok_or(SumcheckError::InvalidProductDimensions)?;
    let word = words
        .get(row / u64::BITS as usize)
        .ok_or(SumcheckError::InvalidProductDimensions)?;
    Ok((word >> (row % u64::BITS as usize)) & 1)
}

pub(super) fn validate_shared_constant(rows: &[Vec<u64>]) -> Result<(), ProtocolError> {
    let packed = rows
        .get(SHA256_SHARED_CONSTANT_CELL)
        .ok_or(ProtocolError::InvalidGeometry)?;
    if packed.first().is_none_or(|word| word & 1 == 0) {
        return Err(ProtocolError::InvalidSharedConstant);
    }
    Ok(())
}

fn validate_public_statement(
    expected: usize,
    public_statement: &[Sha256CompressionStatement],
) -> Result<(), ProtocolError> {
    if public_statement.len() != expected {
        return Err(ProtocolError::InvalidPublicStatementLength {
            expected,
            actual: public_statement.len(),
        });
    }
    Ok(())
}

fn assignment_binding(
    prepared: &PreparedSha256CompressionBatch,
    commitment: &Commitment,
    config: &dyn LigeritoStatementConfig,
    public_statement_binding: &[u8; 32],
) -> Result<[u8; 32], ProtocolError> {
    let mut hash = Hasher::new();
    hash.update(SHA256_ASSIGNMENT_BINDING_DOMAIN);
    hash.update(&commitment.root);
    hash.update(prepared.integer_relation_digest());
    hash.update(&prepared.map().digest());
    match (prepared.product_map(), prepared.product_assignment_params()) {
        (Some(product_map), Some(product_params)) => {
            hash.update(&[1]);
            hash.update(&product_map.digest());
            for value in [
                product_params.row_vars,
                product_params.col_vars,
                product_params.word_bits,
            ] {
                hash_usize(&mut hash, value)?;
            }
        }
        (None, None) => {
            hash.update(&[0]);
        }
        _ => return Err(ProtocolError::InvalidGeometry),
    }
    hash.update(public_statement_binding);
    for value in [
        commitment.params.m,
        commitment.params.log_inv_rate,
        commitment.params.log_batch_size,
        prepared.instances(),
        prepared.log_instance_capacity(),
        prepared.assignment_params().row_vars,
        prepared.assignment_params().col_vars,
        prepared.assignment_params().word_bits,
        prepared.source_params().row_vars,
        prepared.source_params().col_vars,
        prepared.source_params().word_bits,
    ] {
        hash_usize(&mut hash, value)?;
    }
    hash.update(&[profile_code(commitment.params.profile)]);
    hash.update(&[hash_code(commitment.params.merkle_hash)]);
    hash_security_profile(&mut hash, prepared)?;
    hash_ligerito_config(&mut hash, config)?;
    Ok(*hash.finalize().as_bytes())
}

fn validate_ligerito_config_for(
    prepared: &PreparedSha256CompressionBatch,
    actual: &dyn LigeritoStatementConfig,
) -> Result<(), ProtocolError> {
    let (expected, _) = sha256_compression_configs(prepared)?;
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

fn hash_security_profile(
    hash: &mut Hasher,
    prepared: &PreparedSha256CompressionBatch,
) -> Result<(), ProtocolError> {
    hash_security_params(hash, prepared.security())
}

/// Binds every instantiated security parameter (the profile name, target,
/// intervals, grinding schedule, and Ligerito target).
pub(super) fn hash_security_params(
    hash: &mut Hasher,
    security: &super::super::profile::IopSecurityParams,
) -> Result<(), ProtocolError> {
    hash_usize(hash, security.profile_name.len())?;
    hash.update(security.profile_name.as_bytes());
    hash.update(&security.lambda.to_le_bytes());
    hash.update(&security.projection_min.to_le_bytes());
    hash.update(&security.projection_max.to_le_bytes());
    hash.update(&[u8::from(security.projection_full_width)]);
    hash.update(&security.initial_grinding_bits.to_le_bytes());
    hash.update(&security.piop_round_grinding_bits.to_le_bytes());
    hash.update(&security.terminal_grinding_bits.to_le_bytes());
    match security.reduction {
        Some(reduction) => {
            hash.update(&[1]);
            hash.update(&reduction.min.to_le_bytes());
            hash.update(&reduction.max.to_le_bytes());
            hash.update(&reduction.grinding_bits.to_le_bytes());
        }
        None => {
            hash.update(&[0]);
        }
    }
    hash.update(&security.forest_round_grinding_bits.to_le_bytes());
    hash.update(&security.ring_switch_grinding_bits.to_le_bytes());
    if let Some(ood) = security.ood {
        // Present only when Round 0 (the out-of-domain sample) runs, so
        // Round-0-less statements keep their digest.
        hash.update(&[1]);
        hash.update(&ood.grinding_bits.to_le_bytes());
    }
    hash_usize(hash, security.ligerito_target_bits)?;
    Ok(())
}

pub(super) fn hash_ligerito_config(
    hash: &mut Hasher,
    config: &dyn LigeritoStatementConfig,
) -> Result<(), ProtocolError> {
    for value in [
        config.recursive_steps(),
        config.initial_log_msg_cols(),
        config.initial_log_num_interleaved(),
        config.initial_k(),
    ] {
        hash_usize(hash, value)?;
    }
    for values in [
        config.log_inv_rates(),
        config.recursive_log_msg_cols(),
        config.recursive_ks(),
        config.queries(),
        config.grinding_bits(),
        config.fold_grinding_bits(),
        config.ood_samples(),
    ] {
        hash_usize(hash, values.len())?;
        for &value in values {
            hash_usize(hash, value)?;
        }
    }
    hash.update(&[hash_code(config.merkle_hash())]);
    Ok(())
}

fn public_statement_binding(
    public_statement: &[Sha256CompressionStatement],
) -> Result<[u8; 32], ProtocolError> {
    let mut hash = Hasher::new();
    hash.update(SHA256_PUBLIC_STATEMENT_DOMAIN);
    hash_usize(&mut hash, public_statement.len())?;
    for statement in public_statement {
        for word in statement.words() {
            hash.update(&word.to_le_bytes());
        }
    }
    Ok(*hash.finalize().as_bytes())
}

pub(super) fn hash_usize(hash: &mut Hasher, value: usize) -> Result<(), ProtocolError> {
    hash.update(
        &u64::try_from(value)
            .map_err(|_| ProtocolError::BindingEncodingOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {

    use crate::{
        pcs::FQ_MOD,
        piop::spartan::sha256::{
            generate_sha256_compression_witnesses, prepare_sha256_compression_batch,
            prepare_sha256_compression_batch_for_assignment_rows,
        },
        transcript::{
            Blake3Transcript,
            traits::{ConstTranscribable, Transcript},
        },
    };

    use super::*;

    struct RecordingTranscript {
        inner: Blake3Transcript,
        pre_challenge_bytes: Vec<u8>,
        challenge_seen: bool,
    }

    impl RecordingTranscript {
        fn new() -> Self {
            Self {
                inner: Blake3Transcript::new(),
                pre_challenge_bytes: Vec::new(),
                challenge_seen: false,
            }
        }

        fn absorbed_before_first_challenge(&self, needle: &[u8]) -> bool {
            self.pre_challenge_bytes
                .windows(needle.len())
                .any(|window| window == needle)
        }
    }

    impl Transcript for RecordingTranscript {
        fn begin_sampling(&mut self) {
            self.inner.begin_sampling();
        }
        fn fill_sampling_bytes(&mut self, output: &mut [u8]) {
            self.challenge_seen = true;
            self.inner.fill_sampling_bytes(output);
        }

        fn get_challenge<T: ConstTranscribable>(&mut self) -> T {
            self.challenge_seen = true;
            self.inner.get_challenge()
        }

        fn absorb_inner(&mut self, value: &[u8]) {
            if !self.challenge_seen {
                self.pre_challenge_bytes.extend_from_slice(value);
            }
            self.inner.absorb_inner(value);
        }
    }

    fn assert_nonce_mutation_rejects(
        prepared: &PreparedSha256CompressionBatch,
        public_statement: &[Sha256CompressionStatement],
        commitment: &Commitment,
        proof: &LinearProof,
        vc: &LigVerifierConfig,
        mutate: impl Fn(&mut LinearProof, u64),
    ) {
        let rejected = (1..=32_u64).any(|delta| {
            let mut tampered = proof.clone();
            mutate(&mut tampered, delta);
            let mut transcript = Blake3Transcript::new();
            verify_sha256_compressions_with_config(
                &mut transcript,
                prepared,
                public_statement,
                commitment,
                &tampered,
                vc,
            )
            .is_err()
        });
        assert!(rejected, "at least one modified grinding nonce must reject");
    }

    fn input(instance: usize) -> super::super::witness::Sha256CompressionInput {
        let mut state = [0u32; 8];
        let mut block = [0u32; 16];
        for (word, value) in state.iter_mut().enumerate() {
            *value = (instance as u32)
                .wrapping_mul(0x9e37_79b9)
                .rotate_left(word as u32);
        }
        for (word, value) in block.iter_mut().enumerate() {
            *value = (instance as u32 ^ word as u32)
                .wrapping_mul(0x85eb_ca6b)
                .rotate_right(word as u32);
        }
        (state, block)
    }

    fn public_statements(
        inputs: &[super::super::witness::Sha256CompressionInput],
        outputs: &[[u32; 8]],
    ) -> Vec<Sha256CompressionStatement> {
        inputs
            .iter()
            .copied()
            .zip(outputs.iter().copied())
            .map(|(input, output)| Sha256CompressionStatement::new(input, output))
            .collect()
    }

    #[test]
    fn ligerito_profiles_cover_every_supported_sha_batch() {
        for log_compressions in 7..=super::super::prime::SHA256_MAX_LOG_COMPRESSIONS {
            let prepared = prepare_sha256_compression_batch(log_compressions).unwrap();
            let (pc, vc) = sha256_compression_configs(&prepared).unwrap();
            assert_eq!(pc.merkle_hash, flock_core::merkle::HashKind::Blake3);
            assert_eq!(pc.log_inv_rates, vc.log_inv_rates);
            assert_eq!(pc.queries, vc.queries);
            assert_eq!(pc.fold_grinding_bits, vc.fold_grinding_bits);
            assert_eq!(pc.fold_grinding_bits.len(), pc.recursive_steps + 1);

            let profile = Sha256PrimeProfile::from_security(
                prepared.security(),
                prepared.log_instance_capacity(),
            )
            .unwrap();
            assert!(u128::from(*prepared.max_boolean_residual_bound()) < profile.min_prime);
        }

        let by_rows = prepare_sha256_compression_batch_for_assignment_rows(21).unwrap();
        assert_eq!(by_rows.instances(), 102);
        assert_eq!(by_rows.log_instance_capacity(), 7);
        assert_eq!(by_rows.assignment_params().cells(), 1 << 21);
        sha256_compression_configs(&by_rows).unwrap();
        Sha256PrimeProfile::from_security(by_rows.security(), by_rows.log_instance_capacity())
            .unwrap();
    }

    #[test]
    fn production_boundary_and_both_regimes_bind_public_outputs() {
        // These algebraic shapes have fewer than the supported 2^20 committed bits.
        for exponent in 4..=6 {
            assert!(prepare_sha256_compression_batch(exponent).is_err());
        }
        let exponent = 7;
        for selection in [
            crate::ligerito_flock::LigeritoSelection::JOHNSON,
            crate::ligerito_flock::LigeritoSelection::MATCHED_UDR,
        ] {
            let prepared = prepare_sha256_compression_batch(exponent)
                .unwrap()
                .with_ligerito(selection)
                .unwrap();
            assert!(prepared.product_assignment_params().is_some());
            assert!(prepared.opening_params().row_vars >= LOG_PACKING);
            assert!(prepared.security().accounting.achieved_bits() >= 100.0);
            let inputs = (0..prepared.instances()).map(input).collect::<Vec<_>>();
            let witness = generate_sha256_compression_witnesses(&prepared, &inputs).unwrap();
            let mut statements = public_statements(&inputs, witness.outputs());
            let (pc, vc) = sha256_compression_configs(&prepared).unwrap();
            let hint =
                commit_sha256_compression_witness_with_config(&prepared, &witness, &pc).unwrap();
            let proof = prove_sha256_compressions_with_config(
                &mut Blake3Transcript::new(),
                &prepared,
                &statements,
                &witness,
                &hint,
                &pc,
            )
            .unwrap();
            assert_eq!(
                proof.bitz().mfs.len(),
                2,
                "production product layout forests"
            );
            verify_sha256_compressions_with_config(
                &mut Blake3Transcript::new(),
                &prepared,
                &statements,
                &hint.commitment,
                &proof,
                &vc,
            )
            .unwrap();

            statements.last_mut().unwrap().claimed_output[7] ^= 1 << 31;
            assert!(
                verify_sha256_compressions_with_config(
                    &mut Blake3Transcript::new(),
                    &prepared,
                    &statements,
                    &hint.commitment,
                    &proof,
                    &vc,
                )
                .is_err(),
                "the final output bit must be bound at 2^{exponent} compressions"
            );
        }
    }

    #[test]
    fn assignment_row_sized_non_power_batch_roundtrip_and_public_binding() {
        let prepared = prepare_sha256_compression_batch_for_assignment_rows(21).unwrap();
        assert_eq!(prepared.instances(), 102);

        let inputs = (0..prepared.instances()).map(input).collect::<Vec<_>>();
        let witness = generate_sha256_compression_witnesses(&prepared, &inputs).unwrap();
        let public_statement = public_statements(&inputs, witness.outputs());
        let (pc, vc) = sha256_compression_configs(&prepared).unwrap();
        let hint = commit_sha256_compression_witness_with_config(&prepared, &witness, &pc).unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        let proof = prove_sha256_compressions_with_config(
            &mut prover_transcript,
            &prepared,
            &public_statement,
            &witness,
            &hint,
            &pc,
        )
        .unwrap();

        let mut verifier_transcript = Blake3Transcript::new();
        verify_sha256_compressions_with_config(
            &mut verifier_transcript,
            &prepared,
            &public_statement,
            &hint.commitment,
            &proof,
            &vc,
        )
        .unwrap();

        let mut false_statement = public_statement.clone();
        false_statement.last_mut().unwrap().claimed_output[7] ^= 1 << 31;
        let mut false_transcript = Blake3Transcript::new();
        assert!(
            verify_sha256_compressions_with_config(
                &mut false_transcript,
                &prepared,
                &false_statement,
                &hint.commitment,
                &proof,
                &vc,
            )
            .is_err(),
            "the last public output bit of the partial final repetition must be bound",
        );
    }

    #[test]
    fn explicit_inner_sumcheck_layout_roundtrips_across_forest_counts() {
        use super::super::super::{
            Sha256OpeningLayout, prepare_sha256_compression_batch_with_profile_and_layout,
        };
        use crate::piop::spartan::profile::Lambda100;
        // 2^7 compressions: 2^22 assignment cells. t = 13 keeps one forest
        // (113-bit primes, c_w = 113); t = 15 narrows c_w to 111 and needs two.
        const LOG_COMPRESSIONS: usize = 7;
        let inputs = (0..1usize << LOG_COMPRESSIONS)
            .map(input)
            .collect::<Vec<_>>();
        for (row_vars, expected_forests) in [(13usize, 1usize), (15, 2)] {
            let prepared = prepare_sha256_compression_batch_with_profile_and_layout::<Lambda100>(
                LOG_COMPRESSIONS,
                Sha256OpeningLayout::InnerSumcheck { row_vars },
            )
            .unwrap();
            assert_eq!(
                prepared.opening_layout(),
                Sha256OpeningLayout::InnerSumcheck { row_vars }
            );
            assert!(prepared.product_assignment_params().is_none());
            let h_layout = *prepared.opening_params();
            assert_eq!(
                (h_layout.row_vars, h_layout.row_vars + h_layout.col_vars),
                (row_vars, 22)
            );
            let witness = generate_sha256_compression_witnesses(&prepared, &inputs).unwrap();
            let public_statement = public_statements(&inputs, witness.outputs());
            let (pc, vc) = sha256_compression_configs(&prepared).unwrap();
            let hint =
                commit_sha256_compression_witness_with_config(&prepared, &witness, &pc).unwrap();
            let mut prover_transcript = Blake3Transcript::new();
            let proof = prove_sha256_compressions_with_config(
                &mut prover_transcript,
                &prepared,
                &public_statement,
                &witness,
                &hint,
                &pc,
            )
            .unwrap();
            assert_eq!(
                proof.bitz().mfs.len(),
                expected_forests,
                "forests at t={row_vars}"
            );
            assert_eq!(
                proof.inner().round_polynomials.len(),
                22,
                "inner rounds at t={row_vars}"
            );
            let mut verifier_transcript = Blake3Transcript::new();
            verify_sha256_compressions_with_config(
                &mut verifier_transcript,
                &prepared,
                &public_statement,
                &hint.commitment,
                &proof,
                &vc,
            )
            .unwrap();
        }
        for row_vars in [LOG_PACKING - 1, 22] {
            assert!(matches!(
                prepare_sha256_compression_batch_with_profile_and_layout::<Lambda100>(
                    LOG_COMPRESSIONS,
                    Sha256OpeningLayout::InnerSumcheck { row_vars },
                ),
                Err(crate::piop::spartan::Sha256ConstraintError::InvalidOpeningRowVars { .. })
            ));
        }
    }

    #[test]
    fn transposed_product_layout_roundtrips_without_an_inner_sumcheck() {
        use super::super::super::{
            Sha256OpeningLayout, prepare_sha256_compression_batch_with_profile_and_layout,
        };
        use crate::piop::spartan::profile::Lambda100;
        // 2^7 compressions: 2^22 assignment cells, local stride 2^15. Every
        // admissible split has t >= 15, so two forests at 113-bit primes.
        const LOG_COMPRESSIONS: usize = 7;
        let inputs = (0..1usize << LOG_COMPRESSIONS)
            .map(input)
            .collect::<Vec<_>>();
        for row_vars in [15usize, 17, 22] {
            let prepared = prepare_sha256_compression_batch_with_profile_and_layout::<Lambda100>(
                LOG_COMPRESSIONS,
                Sha256OpeningLayout::ProductTransposed { row_vars },
            )
            .unwrap();
            let product_map = prepared.product_map().unwrap();
            assert_eq!(product_map.order(), PackedSourceOrder::InstanceMajor);
            let h_layout = *prepared.opening_params();
            assert_eq!(
                (h_layout.row_vars, h_layout.row_vars + h_layout.col_vars),
                (row_vars, 22)
            );
            let witness = generate_sha256_compression_witnesses(&prepared, &inputs).unwrap();
            let public_statement = public_statements(&inputs, witness.outputs());
            let (pc, vc) = sha256_compression_configs(&prepared).unwrap();
            let hint =
                commit_sha256_compression_witness_with_config(&prepared, &witness, &pc).unwrap();
            let mut prover_transcript = Blake3Transcript::new();
            let proof = prove_sha256_compressions_with_config(
                &mut prover_transcript,
                &prepared,
                &public_statement,
                &witness,
                &hint,
                &pc,
            )
            .unwrap();
            assert_eq!(proof.bitz().mfs.len(), 2, "forests at t={row_vars}");
            assert!(proof.inner().round_polynomials.is_empty());
            assert_eq!(proof.bitz().us[0].len(), 1 << (22 - row_vars));
            let mut verifier_transcript = Blake3Transcript::new();
            verify_sha256_compressions_with_config(
                &mut verifier_transcript,
                &prepared,
                &public_statement,
                &hint.commitment,
                &proof,
                &vc,
            )
            .unwrap();
        }
        for row_vars in [14usize, 23] {
            assert!(matches!(
                prepare_sha256_compression_batch_with_profile_and_layout::<Lambda100>(
                    LOG_COMPRESSIONS,
                    Sha256OpeningLayout::ProductTransposed { row_vars },
                ),
                Err(crate::piop::spartan::Sha256ConstraintError::InvalidOpeningRowVars { .. })
            ));
        }
    }

    #[test]
    fn runtime_prime_roundtrip() {
        // Pinned to the grinded reference schedule: this test exercises the
        // per-round and initial/terminal grinding machinery, which the
        // λ = 100 default profile deliberately skips.
        const LOG_COMPRESSIONS: usize = 7;
        let prepared = super::super::super::prepare_sha256_compression_batch_with_profile::<
            crate::piop::spartan::profile::Sha128ReferenceSchedule,
        >(LOG_COMPRESSIONS)
        .unwrap();
        let inputs = (0..1usize << LOG_COMPRESSIONS)
            .map(input)
            .collect::<Vec<_>>();
        let witness = generate_sha256_compression_witnesses(&prepared, &inputs).unwrap();
        let public_statement = public_statements(&inputs, witness.outputs());
        let (pc, vc) = sha256_compression_configs(&prepared).unwrap();
        let hint = commit_sha256_compression_witness_with_config(&prepared, &witness, &pc).unwrap();

        let public_binding = public_statement_binding(&public_statement).unwrap();
        let expected_assignment_binding =
            assignment_binding(&prepared, &hint.commitment, &pc, &public_binding).unwrap();

        let short_statement = &public_statement[..public_statement.len() - 1];
        let mut short_prover_transcript = Blake3Transcript::new();
        let mut untouched = short_prover_transcript.clone();
        assert!(matches!(
            prove_sha256_compressions_with_config(
                &mut short_prover_transcript,
                &prepared,
                short_statement,
                &witness,
                &hint,
                &pc,
            ),
            Err(ProtocolError::InvalidPublicStatementLength {
                expected: 128,
                actual: 127
            })
        ));
        assert_eq!(
            short_prover_transcript.get_challenge::<u128>(),
            untouched.get_challenge::<u128>(),
            "preflight rejection must not mutate the transcript"
        );

        let mut invalid_prefix_transcript = Blake3Transcript::new();
        let mut untouched = invalid_prefix_transcript.clone();
        assert!(matches!(
            prove_sha256_compressions_with_prefix_vars_and_config(
                &mut invalid_prefix_transcript,
                &prepared,
                &public_statement,
                &witness,
                &hint,
                SHA256_INNER_PREFIX_MAX_VARS + 1,
                &pc,
            ),
            Err(ProtocolError::InvalidInnerPrefix { actual: 5, max: 4 })
        ));
        assert_eq!(
            invalid_prefix_transcript.get_challenge::<u128>(),
            untouched.get_challenge::<u128>(),
            "invalid prover-local K must reject before transcript absorption"
        );

        let default_prepared = prepare_sha256_compression_batch(LOG_COMPRESSIONS).unwrap();
        let (wrong_pc, wrong_vc) = sha256_compression_configs(&default_prepared).unwrap();
        assert!(matches!(
            commit_sha256_compression_witness_with_config(&prepared, &witness, &wrong_pc),
            Err(ProtocolError::MismatchedLigeritoConfig)
        ));
        let mut wrong_config_transcript = Blake3Transcript::new();
        assert!(matches!(
            prove_sha256_compressions_with_config(
                &mut wrong_config_transcript,
                &prepared,
                &public_statement,
                &witness,
                &hint,
                &wrong_pc,
            ),
            Err(ProtocolError::MismatchedLigeritoConfig)
        ));

        let mut missing_constant = witness.clone();
        missing_constant.source_rows_mut_for_tests()[0][0] &= !1;
        assert!(matches!(
            commit_sha256_compression_witness_with_config(&prepared, &missing_constant, &pc),
            Err(ProtocolError::InvalidSharedConstant)
        ));
        let bypass_hint = commit_rs_ligerito_rows(
            prepared.source_params(),
            missing_constant.source_rows().to_vec(),
            &pc,
        );
        let mut bypass_transcript = Blake3Transcript::new();
        let mut untouched = bypass_transcript.clone();
        assert!(matches!(
            prove_sha256_compressions_with_config(
                &mut bypass_transcript,
                &prepared,
                &public_statement,
                &missing_constant,
                &bypass_hint,
                &pc,
            ),
            Err(ProtocolError::InvalidSharedConstant)
        ));
        assert_eq!(
            bypass_transcript.get_challenge::<u128>(),
            untouched.get_challenge::<u128>(),
            "malformed witnesses must reject before transcript absorption"
        );

        let mut prover_transcript = RecordingTranscript::new();
        let proof = prove_sha256_compressions_with_config(
            &mut prover_transcript,
            &prepared,
            &public_statement,
            &witness,
            &hint,
            &pc,
        )
        .unwrap();
        assert!(
            prover_transcript.absorbed_before_first_challenge(&expected_assignment_binding),
            "the commitment, public statement, profile, and PCS config must be bound before q"
        );
        assert!(proof.inner().round_polynomials.is_empty());
        assert!(proof.inner_nonces().is_empty());

        let mut verifier_transcript = Blake3Transcript::new();
        verify_sha256_compressions_with_config(
            &mut verifier_transcript,
            &prepared,
            &public_statement,
            &hint.commitment,
            &proof,
            &vc,
        )
        .unwrap();

        let mut tampered_inner = proof.clone();
        let field_config = Fp::<2>::make_cfg(&Uint::from(FQ_MOD)).expect("fixed test field");
        let zero = SpartanBitzField::zero_with_cfg(&field_config);
        tampered_inner
            .inner_mut()
            .round_polynomials
            .push([zero.clone(), zero.clone(), zero]);
        let mut tampered_inner_transcript = Blake3Transcript::new();
        assert!(
            verify_sha256_compressions_with_config(
                &mut tampered_inner_transcript,
                &prepared,
                &public_statement,
                &hint.commitment,
                &tampered_inner,
                &vc,
            )
            .is_err()
        );

        let mut short_verifier_transcript = Blake3Transcript::new();
        let mut untouched = short_verifier_transcript.clone();
        assert!(matches!(
            verify_sha256_compressions_with_config(
                &mut short_verifier_transcript,
                &prepared,
                short_statement,
                &hint.commitment,
                &proof,
                &vc,
            ),
            Err(ProtocolError::InvalidPublicStatementLength {
                expected: 128,
                actual: 127
            })
        ));
        assert_eq!(
            short_verifier_transcript.get_challenge::<u128>(),
            untouched.get_challenge::<u128>(),
            "verifier preflight rejection must not mutate the transcript"
        );

        let mut wrong_config_transcript = Blake3Transcript::new();
        assert!(matches!(
            verify_sha256_compressions_with_config(
                &mut wrong_config_transcript,
                &prepared,
                &public_statement,
                &hint.commitment,
                &proof,
                &wrong_vc,
            ),
            Err(ProtocolError::MismatchedLigeritoConfig)
        ));

        assert_nonce_mutation_rejects(
            &prepared,
            &public_statement,
            &hint.commitment,
            &proof,
            &vc,
            |proof, delta| {
                *proof.initial_nonce_mut() = proof.initial_nonce().wrapping_add(delta);
            },
        );
        assert_nonce_mutation_rejects(
            &prepared,
            &public_statement,
            &hint.commitment,
            &proof,
            &vc,
            |proof, delta| {
                *proof.terminal_nonce_mut() = proof.terminal_nonce().wrapping_add(delta);
            },
        );

        let mut wrong_commitment = hint.commitment.clone();
        wrong_commitment.root[0] ^= 1;
        let mut wrong_commitment_transcript = Blake3Transcript::new();
        assert!(
            verify_sha256_compressions_with_config(
                &mut wrong_commitment_transcript,
                &prepared,
                &public_statement,
                &wrong_commitment,
                &proof,
                &vc,
            )
            .is_err()
        );

        let mut false_statement = public_statement.clone();
        false_statement[3].claimed_output[0] ^= 1;
        let mut false_transcript = Blake3Transcript::new();
        assert!(
            verify_sha256_compressions_with_config(
                &mut false_transcript,
                &prepared,
                &false_statement,
                &hint.commitment,
                &proof,
                &vc,
            )
            .is_err()
        );
    }

    #[test]
    fn runtime_prime_rejects_shared_false_public_components() {
        const LOG_COMPRESSIONS: usize = 7;
        const ATTACKED_INSTANCE: usize = 5;

        let prepared = prepare_sha256_compression_batch(LOG_COMPRESSIONS).unwrap();
        let inputs = (0..1usize << LOG_COMPRESSIONS)
            .map(input)
            .collect::<Vec<_>>();
        let witness = generate_sha256_compression_witnesses(&prepared, &inputs).unwrap();
        let public_statement = public_statements(&inputs, witness.outputs());
        let (pc, vc) = sha256_compression_configs(&prepared).unwrap();
        let hint = commit_sha256_compression_witness_with_config(&prepared, &witness, &pc).unwrap();

        // Prover and verifier deliberately agree on each false statement, so
        // rejection cannot come merely from transcript divergence: the
        // committed public-wire equality itself must fail.
        for component in 0..3 {
            let mut false_statement = public_statement.clone();
            match component {
                0 => false_statement[ATTACKED_INSTANCE].state[0] ^= 1,
                1 => false_statement[ATTACKED_INSTANCE].block[0] ^= 1,
                2 => false_statement[ATTACKED_INSTANCE].claimed_output[0] ^= 1,
                _ => unreachable!(),
            }

            let mut prover_transcript = Blake3Transcript::new();
            let proof = prove_sha256_compressions_with_config(
                &mut prover_transcript,
                &prepared,
                &false_statement,
                &witness,
                &hint,
                &pc,
            )
            .expect("a false public claim still produces a candidate proof");

            let mut verifier_transcript = Blake3Transcript::new();
            assert!(
                verify_sha256_compressions_with_config(
                    &mut verifier_transcript,
                    &prepared,
                    &false_statement,
                    &hint.commitment,
                    &proof,
                    &vc,
                )
                .is_err()
            );
        }

        // Cell 769 is the first non-input wire in the first local assignment.
        // Changing it while retaining the source commitment must not produce
        // an accepting linear proof.
        const DERIVED_ASSIGNMENT_CELL: usize = 769;
        let mut inconsistent_witness = witness.clone();
        let product_p_h = prepared.product_assignment_params().unwrap();
        let flat_cell = DERIVED_ASSIGNMENT_CELL * prepared.instances();
        let column = flat_cell >> product_p_h.row_vars;
        let row = flat_cell & (product_p_h.rows() - 1);
        inconsistent_witness
            .product_assignment_rows_mut_for_tests()
            .unwrap()[column][row / 64] ^= 1 << (row % 64);
        let mut inconsistent_prover_transcript = Blake3Transcript::new();
        if let Ok(inconsistent_proof) = prove_sha256_compressions_with_config(
            &mut inconsistent_prover_transcript,
            &prepared,
            &public_statement,
            &inconsistent_witness,
            &hint,
            &pc,
        ) {
            let mut inconsistent_verifier_transcript = Blake3Transcript::new();
            assert!(
                verify_sha256_compressions_with_config(
                    &mut inconsistent_verifier_transcript,
                    &prepared,
                    &public_statement,
                    &hint.commitment,
                    &inconsistent_proof,
                    &vc,
                )
                .is_err()
            );
        }

        // Flip one committed, non-public hint bit and update both assignment
        // views by the public Bit map. This preserves h = M f exactly, so
        // rejection must come from the batched C h = 0 constraints rather
        // than from an inconsistent uncommitted h oracle.
        const PRIVATE_LOCAL_SOURCE_CELL: usize = 1_000;
        assert!((0..SHA256_PUBLIC_WORDS).all(|word_slot| {
            (0..SHA256_PUBLIC_WORD_BITS)
                .all(|bit| sha256_public_f_column(word_slot, bit) != PRIVATE_LOCAL_SOURCE_CELL)
        }));
        let mut local_residuals =
            vec![num_bigint::BigInt::default(); prepared.linear_relation().row_count()];
        for local_h in prepared
            .map()
            .local()
            .matrix()
            .column(PRIVATE_LOCAL_SOURCE_CELL)
            .unwrap()
            .indices()
        {
            for (row, coefficient) in prepared
                .linear_relation()
                .matrix()
                .column(*local_h)
                .unwrap()
            {
                local_residuals[row] += *coefficient;
            }
        }
        assert!(
            local_residuals
                .iter()
                .any(|residual| residual != &num_bigint::BigInt::default())
        );

        let source_cell = PRIVATE_LOCAL_SOURCE_CELL;
        let canonical_h_cells = prepared
            .map()
            .column_rows(source_cell)
            .unwrap()
            .collect::<Vec<_>>();
        let product_h_cells = prepared
            .product_map()
            .unwrap()
            .column_rows(source_cell)
            .unwrap()
            .collect::<Vec<_>>();
        let mut constraint_attack = witness.clone();
        let f_layout = prepared.source_params();
        let source_column = source_cell >> f_layout.row_vars;
        let source_row = source_cell & (f_layout.rows() - 1);
        constraint_attack.source_rows_mut_for_tests()[source_column][source_row / 64] ^=
            1 << (source_row % 64);
        for flat_cell in canonical_h_cells {
            let canonical_p_h = prepared.assignment_params();
            let column = flat_cell >> canonical_p_h.row_vars;
            let row = flat_cell & (canonical_p_h.rows() - 1);
            constraint_attack.assignment_rows_mut_for_tests()[column][row / 64] ^= 1 << (row % 64);
        }
        for flat_cell in product_h_cells {
            let column = flat_cell >> product_p_h.row_vars;
            let row = flat_cell & (product_p_h.rows() - 1);
            constraint_attack
                .product_assignment_rows_mut_for_tests()
                .unwrap()[column][row / 64] ^= 1 << (row % 64);
        }
        let attack_hint =
            commit_sha256_compression_witness_with_config(&prepared, &constraint_attack, &pc)
                .unwrap();
        let mut attack_prover_transcript = Blake3Transcript::new();
        let attack_proof = prove_sha256_compressions_with_config(
            &mut attack_prover_transcript,
            &prepared,
            &public_statement,
            &constraint_attack,
            &attack_hint,
            &pc,
        )
        .unwrap();
        let mut attack_verifier_transcript = Blake3Transcript::new();
        assert!(
            verify_sha256_compressions_with_config(
                &mut attack_verifier_transcript,
                &prepared,
                &public_statement,
                &attack_hint.commitment,
                &attack_proof,
                &vc,
            )
            .is_err()
        );
    }

    #[test]
    fn local_rows_product_layout_rejects_a_false_public_output() {
        const LOG_COMPRESSIONS: usize = 7;
        let prepared =
            super::super::constraints::prepare_sha256_compression_batch_for_product_t_test(
                LOG_COMPRESSIONS,
                15,
            )
            .unwrap()
            .with_ligerito(crate::ligerito_flock::LigeritoSelection::JOHNSON)
            .unwrap();
        assert_eq!(prepared.product_layout_name(), Some("local_rows"));

        let inputs = (0..1usize << LOG_COMPRESSIONS)
            .map(input)
            .collect::<Vec<_>>();
        let witness = generate_sha256_compression_witnesses(&prepared, &inputs).unwrap();
        let mut false_statement = public_statements(&inputs, witness.outputs());
        false_statement[5].claimed_output[0] ^= 1;
        let (pc, vc) = sha256_compression_configs(&prepared).unwrap();
        let hint = commit_sha256_compression_witness_with_config(&prepared, &witness, &pc).unwrap();

        // Both sides bind the same false statement. Rejection therefore comes
        // from the committed-wire claim, not from transcript divergence.
        let mut prover_transcript = Blake3Transcript::new();
        let proof = prove_sha256_compressions_with_config(
            &mut prover_transcript,
            &prepared,
            &false_statement,
            &witness,
            &hint,
            &pc,
        )
        .expect("a false public claim still produces a candidate proof");
        let mut verifier_transcript = Blake3Transcript::new();
        assert!(
            verify_sha256_compressions_with_config(
                &mut verifier_transcript,
                &prepared,
                &false_statement,
                &hint.commitment,
                &proof,
                &vc,
            )
            .is_err()
        );
    }

    #[test]
    fn public_statement_binding_covers_every_public_component_and_instance_order() {
        let inputs = [input(1), input(2)];
        let outputs = [[3_u32; 8], [4_u32; 8]];
        let statement = public_statements(&inputs, &outputs);
        let binding = public_statement_binding(&statement).unwrap();

        let mut changed_state = statement.clone();
        changed_state[0].state[0] ^= 1;
        assert_ne!(public_statement_binding(&changed_state).unwrap(), binding);

        let mut changed_block = statement.clone();
        changed_block[0].block[0] ^= 1;
        assert_ne!(public_statement_binding(&changed_block).unwrap(), binding);

        let mut changed_output = statement.clone();
        changed_output[0].claimed_output[0] ^= 1;
        assert_ne!(public_statement_binding(&changed_output).unwrap(), binding);

        let mut swapped = statement.clone();
        swapped.swap(0, 1);
        assert_ne!(public_statement_binding(&swapped).unwrap(), binding);
    }

    #[test]
    fn public_statement_length_validation_rejects_mismatched_count() {
        assert!(matches!(
            validate_public_statement(1, &[]),
            Err(ProtocolError::InvalidPublicStatementLength {
                expected: 1,
                actual: 0
            })
        ));
    }

    #[test]
    fn compact_equality_table_matches_field_table() {
        let field_config = Fp::<2>::make_cfg(&Uint::from(FQ_MOD)).expect("fixed test field");
        let point = (0..12)
            .map(|coordinate| {
                SpartanBitzField::from_with_cfg((coordinate as u64) + 7, &field_config)
            })
            .collect::<Vec<_>>();
        let dense = eq_table(&point, &field_config).unwrap();
        let compact = compact_eq_table(&point, &field_config).unwrap();
        let factored = FactoredEqualityWeights::new(&point, 7, &field_config).unwrap();
        let scale = SpartanBitzField::from_with_cfg(37_u64, &field_config);
        let mut scaled_factored = FactoredEqualityWeights::new(&point, 7, &field_config).unwrap();
        scaled_factored.scale(&scale, &field_config);

        assert_eq!(compact.len(), dense.len());
        assert_eq!(factored.len(), dense.len());
        for (index, (raw, expected)) in compact.into_iter().zip(dense).enumerate() {
            assert_eq!(field_from_raw(raw, &field_config), expected);
            assert_eq!(
                factored.evaluate(index, &field_config),
                Some(expected.clone())
            );
            assert_eq!(
                scaled_factored.evaluate(index, &field_config),
                Some(field_config.mul(&(expected), &(&scale)))
            );
        }
    }

    #[test]
    fn local_beta_matches_manual_c_transpose_eq8_collapse() {
        let prepared = super::super::constraints::prepare_sha256_compression_batch_for_test(0)
            .expect("one-instance test relation");
        let field_config = Fp::<2>::make_cfg(&Uint::from(FQ_MOD)).expect("fixed test field");
        let local_row_point = (0..local_constraint_vars())
            .map(|coordinate| SpartanBitzField::from_with_cfg(coordinate as u64 + 5, &field_config))
            .collect::<Vec<_>>();
        let local_row_weights = eq_table(&local_row_point, &field_config).unwrap();
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_config).unwrap();
        let beta = crate::sumcheck::bridge::repeated::collapse_signed_columns(
            prepared.linear_relation().native_matrix(),
            &local_row_weights,
            &reducer,
        )
        .unwrap();
        let projected = prepared.project_linear_relation(&field_config).unwrap();

        assert_eq!(local_row_weights.len(), 1 << 8);
        assert_eq!(beta.len(), SHA256_H_BAR_LIVE_BITS);
        for (column, expected) in projected.matrix().columns().zip(&beta) {
            let mut manual = SpartanBitzField::zero_with_cfg(&field_config);
            for (local_row, coefficient) in column {
                manual = field_config.add(
                    &(manual),
                    &(&(field_config
                        .mul(&(coefficient.clone()), &(&local_row_weights[local_row])))),
                );
            }
            assert_eq!(&manual, expected);
        }
    }

    #[test]
    fn product_linear_batching_matches_the_gap_free_witness_and_terminal_evaluation() {
        let prepared = super::super::constraints::prepare_sha256_compression_batch_for_test(1)
            .expect("two-instance test relation");
        let inputs = [input(5), input(9)];
        let witness = generate_sha256_compression_witnesses(&prepared, &inputs).unwrap();
        let statements = public_statements(&inputs, witness.outputs());
        let field_config = Fp::<2>::make_cfg(&Uint::from(FQ_MOD)).expect("fixed test field");
        let local_row_point = (0..local_constraint_vars())
            .map(|coordinate| SpartanBitzField::from_with_cfg(coordinate as u64 + 3, &field_config))
            .collect::<Vec<_>>();
        let local_row_weights = eq_table(&local_row_point, &field_config).unwrap();
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_config).unwrap();
        let beta = crate::sumcheck::bridge::repeated::collapse_signed_columns(
            prepared.linear_relation().native_matrix(),
            &local_row_weights,
            &reducer,
        )
        .unwrap();
        let instance_point = [SpartanBitzField::from_with_cfg(13_u64, &field_config)];
        let slot_weights = (0..SHA256_PUBLIC_WORDS * SHA256_PUBLIC_WORD_BITS)
            .map(|slot| SpartanBitzField::from_with_cfg(slot as u64 + 17, &field_config))
            .collect::<Vec<_>>();
        let batching = ProductLinearBatching::new(
            &prepared,
            &statements,
            &instance_point,
            beta,
            slot_weights,
            SpartanBitzField::from_with_cfg(29_u64, &field_config),
            SpartanBitzField::from_with_cfg(31_u64, &field_config),
            &field_config,
        )
        .unwrap();

        let factored = batching.factored_matrix_mle(&field_config).unwrap();
        assert_eq!(
            factored.live_len(),
            prepared.linear_assignment_column_count()
        );
        let mut witness_sum = SpartanBitzField::zero_with_cfg(&field_config);
        for flat_cell in 0..prepared.linear_assignment_column_count() {
            if witness.assignment_bit(flat_cell).unwrap() {
                witness_sum = field_config.add(
                    &(witness_sum),
                    &(&batching.coefficient(flat_cell, &field_config).unwrap()),
                );
            }
        }
        assert_eq!(witness_sum, *batching.initial_claim());

        let product_p_h = prepared.product_assignment_params().unwrap();
        let product_rows = witness.product_assignment_rows().unwrap();
        let order = prepared.product_map().unwrap().order();
        let (row_weights, col_weights, claimed) =
            product_opening_claim(&batching, product_p_h, order, &field_config).unwrap();
        let mut direct_sum = SpartanBitzField::zero_with_cfg(&field_config);
        for flat_cell in 0..product_p_h.cells() {
            if packed_flat_bit(product_rows, product_p_h, flat_cell).unwrap() == 0 {
                continue;
            }
            let row = flat_cell & (product_p_h.rows() - 1);
            let column = flat_cell >> product_p_h.row_vars;
            let mut term = SpartanBitzField::from_with_cfg(
                row_weights.canonical_weight(row, &field_config).unwrap(),
                &field_config,
            );
            term = field_config.mul(
                &(term),
                &(&SpartanBitzField::from_with_cfg(col_weights[column], &field_config)),
            );
            direct_sum = field_config.add(&(direct_sum), &(&term));
        }
        assert_eq!(direct_sum, *batching.initial_claim());
        assert_eq!(
            claimed,
            u128::from(field_config.to_integer(batching.initial_claim()))
        );

        let assignment_point = (0..prepared.assignment_params().row_vars
            + prepared.assignment_params().col_vars)
            .map(|coordinate| SpartanBitzField::from_with_cfg(coordinate as u64 + 37, &field_config))
            .collect::<Vec<_>>();
        let assignment_equality = FactoredEqualityWeights::new(
            &assignment_point,
            prepared.assignment_params().row_vars,
            &field_config,
        )
        .unwrap();
        assert_eq!(
            batching.evaluate(&assignment_point, &field_config).unwrap(),
            batching
                .evaluate_dense(&assignment_equality, &field_config)
                .unwrap()
        );
    }

    #[test]
    fn product_linear_batching_handles_partial_instance_domains_without_inversion() {
        let prepared = prepare_sha256_compression_batch_for_assignment_rows(21)
            .expect("102-instance packed test relation");
        let field_config = Fp::<2>::make_cfg(&Uint::from(FQ_MOD)).expect("fixed test field");
        let instance_point = (0..instance_vars(prepared.instances()).unwrap())
            .map(|coordinate| SpartanBitzField::from_with_cfg(coordinate as u64 + 41, &field_config))
            .collect::<Vec<_>>();
        let instance_weights = eq_table(&instance_point, &field_config).unwrap();
        let active_sum = instance_weights.iter().take(prepared.instances()).fold(
            SpartanBitzField::zero_with_cfg(&field_config),
            |mut sum, weight| {
                sum = field_config.add(&(sum), &(weight));
                sum
            },
        );
        let statements = (0..prepared.instances())
            .map(|instance| Sha256CompressionStatement::new(input(instance), [0; 8]))
            .collect::<Vec<_>>();
        let constant_weight = SpartanBitzField::from_with_cfg(43_u64, &field_config);
        let batching = ProductLinearBatching::new(
            &prepared,
            &statements,
            &instance_point,
            vec![SpartanBitzField::zero_with_cfg(&field_config); SHA256_H_BAR_LIVE_BITS],
            vec![
                SpartanBitzField::zero_with_cfg(&field_config);
                SHA256_PUBLIC_WORDS * SHA256_PUBLIC_WORD_BITS
            ],
            SpartanBitzField::from_with_cfg(47_u64, &field_config),
            constant_weight.clone(),
            &field_config,
        )
        .unwrap();
        let expected = field_config.mul(&(active_sum), &(&constant_weight));

        assert_eq!(
            batching.coefficient(SHA256_SHARED_CONSTANT_CELL, &field_config),
            Ok(expected.clone())
        );
        assert_eq!(*batching.initial_claim(), expected);
        assert_eq!(
            batching
                .factored_matrix_mle(&field_config)
                .unwrap()
                .live_len(),
            prepared.linear_assignment_column_count()
        );
    }

    #[test]
    fn public_linear_batching_matches_the_packed_witness_and_terminal_evaluation() {
        let prepared = super::super::constraints::prepare_sha256_compression_batch_for_test(1)
            .expect("two-instance test relation");
        let inputs = [input(5), input(9)];
        let witness = generate_sha256_compression_witnesses(&prepared, &inputs).unwrap();
        let statements = public_statements(&inputs, witness.outputs());
        let field_config = Fp::<2>::make_cfg(&Uint::from(FQ_MOD)).expect("fixed test field");
        let instance_point = [SpartanBitzField::from_with_cfg(7_u64, &field_config)];
        let slot_weights = (0..SHA256_PUBLIC_WORDS * SHA256_PUBLIC_WORD_BITS)
            .map(|slot| SpartanBitzField::from_with_cfg(slot as u64 + 11, &field_config))
            .collect::<Vec<_>>();
        let public_batch_weight = SpartanBitzField::from_with_cfg(19_u64, &field_config);
        let constant_weight = SpartanBitzField::from_with_cfg(23_u64, &field_config);
        let batching = PublicLinearBatching::new(
            &prepared,
            &statements,
            &instance_point,
            slot_weights,
            public_batch_weight,
            constant_weight,
            &field_config,
        )
        .unwrap();

        let mut witness_sum = SpartanBitzField::zero_with_cfg(&field_config);
        for flat_cell in 0..prepared.assignment_params().cells() {
            if witness.assignment_bit(flat_cell).unwrap() {
                witness_sum = field_config.add(
                    &(witness_sum),
                    &(&batching.coefficient(flat_cell, &field_config).unwrap()),
                );
            }
        }
        assert_eq!(witness_sum, *batching.initial_claim());

        let assignment_point = (0..prepared.assignment_params().row_vars
            + prepared.assignment_params().col_vars)
            .map(|coordinate| SpartanBitzField::from_with_cfg(coordinate as u64 + 29, &field_config))
            .collect::<Vec<_>>();
        let dense_equality = eq_table(&assignment_point, &field_config).unwrap();
        let factored = FactoredEqualityWeights::new(
            &assignment_point,
            prepared.assignment_params().row_vars,
            &field_config,
        )
        .unwrap();
        let mut dense_evaluation = SpartanBitzField::zero_with_cfg(&field_config);
        for (flat_cell, equality) in dense_equality.iter().enumerate() {
            let coefficient = batching.coefficient(flat_cell, &field_config).unwrap();
            dense_evaluation = field_config.add(
                &(dense_evaluation),
                &(&(field_config.mul(&(coefficient), &(equality)))),
            );
        }
        assert_eq!(
            batching.evaluate(&assignment_point, &field_config).unwrap(),
            dense_evaluation
        );
        assert_eq!(
            batching.evaluate_dense(&factored, &field_config).unwrap(),
            dense_evaluation
        );
    }

    #[test]
    fn native_prover_collapse_matches_sparse_verifier_evaluation() {
        let prepared = super::super::constraints::prepare_sha256_compression_batch_for_test(1)
            .expect("two-instance test relation");
        let field_config = Fp::<2>::make_cfg(&Uint::from(FQ_MOD)).expect("fixed test field");
        let relation = prepared
            .project_linear_relation(&field_config)
            .expect("project linear relation");

        let constraint_point = (0..flat_constraint_vars(&prepared).unwrap())
            .map(|coordinate| {
                SpartanBitzField::from_with_cfg((coordinate as u64) + 2, &field_config)
            })
            .collect::<Vec<_>>();
        let constraint_weights = compact_eq_table(&constraint_point, &field_config).unwrap();
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_config).unwrap();
        let collapsed = (0..prepared.assignment_params().cells())
            .map(|flat_column| {
                collapse_flat_linear_column(
                    &prepared,
                    &relation,
                    &constraint_weights,
                    &reducer,
                    flat_column,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let assignment_point = (0..prepared.assignment_params().row_vars
            + prepared.assignment_params().col_vars)
            .map(|coordinate| {
                SpartanBitzField::from_with_cfg((coordinate as u64) + 19, &field_config)
            })
            .collect::<Vec<_>>();
        let assignment_weights = FactoredEqualityWeights::new(
            &assignment_point,
            prepared.assignment_params().row_vars,
            &field_config,
        )
        .unwrap();
        let zero = SpartanBitzField::zero_with_cfg(&field_config);
        let dense_evaluation =
            collapsed
                .iter()
                .enumerate()
                .fold(zero, |mut sum, (flat_column, coefficient)| {
                    let weight = assignment_weights
                        .evaluate(flat_column, &field_config)
                        .unwrap();
                    sum = field_config.add(
                        &(sum),
                        &(&(field_config.mul(&(coefficient.clone()), &(&weight)))),
                    );
                    sum
                });
        let sparse_evaluation = evaluate_flat_linear_collapse_dense(
            &prepared,
            &relation,
            &constraint_weights,
            &assignment_weights,
        )
        .unwrap();
        let repeated_evaluation = evaluate_repeated_flat_linear_collapse(
            &prepared,
            &relation,
            &constraint_point,
            &assignment_point,
        )
        .unwrap();

        assert_eq!(dense_evaluation, sparse_evaluation);
        assert_eq!(dense_evaluation, repeated_evaluation);
    }

    #[test]
    fn affine_equality_repetition_matches_dense_partial_domain() {
        let field_config = Fp::<2>::make_cfg(&Uint::from(FQ_MOD)).expect("fixed test field");
        let left_point = (0..6)
            .map(|coordinate| {
                SpartanBitzField::from_with_cfg((coordinate as u64) + 3, &field_config)
            })
            .collect::<Vec<_>>();
        let right_point = (0..6)
            .map(|coordinate| {
                SpartanBitzField::from_with_cfg((coordinate as u64) + 17, &field_config)
            })
            .collect::<Vec<_>>();
        let instance_point = (0..3)
            .map(|coordinate| {
                SpartanBitzField::from_with_cfg((coordinate as u64) + 31, &field_config)
            })
            .collect::<Vec<_>>();
        let terms = [
            (0, 1, SpartanBitzField::from_with_cfg(5_u64, &field_config)),
            (2, 4, SpartanBitzField::from_with_cfg(7_u64, &field_config)),
            (6, 9, SpartanBitzField::from_with_cfg(11_u64, &field_config)),
        ];
        let optimized = evaluate_affine_equality_repetition(
            5,
            7,
            11,
            &left_point,
            &right_point,
            Some(&instance_point),
            terms.clone(),
            &field_config,
        )
        .unwrap();

        let left_equality = eq_table(&left_point, &field_config).unwrap();
        let right_equality = eq_table(&right_point, &field_config).unwrap();
        let instance_equality = eq_table(&instance_point, &field_config).unwrap();
        let mut dense = SpartanBitzField::zero_with_cfg(&field_config);
        for (instance, instance_weight) in instance_equality.iter().take(5).enumerate() {
            for (left_offset, right_offset, coefficient) in &terms {
                let mut term = field_config.mul(
                    &(coefficient.clone()),
                    &(&left_equality[7 * instance + left_offset]),
                );
                term = field_config.mul(&(term), &(&right_equality[11 * instance + right_offset]));
                term = field_config.mul(&(term), &(instance_weight));
                dense = field_config.add(&(dense), &(&term));
            }
        }
        assert_eq!(optimized, dense);

        let shared_terms = [
            (1, 0, SpartanBitzField::from_with_cfg(13_u64, &field_config)),
            (6, 0, SpartanBitzField::from_with_cfg(17_u64, &field_config)),
        ];
        let optimized_shared = evaluate_affine_equality_repetition(
            5,
            7,
            0,
            &left_point,
            &right_point,
            None,
            shared_terms.clone(),
            &field_config,
        )
        .unwrap();
        let mut dense_shared = SpartanBitzField::zero_with_cfg(&field_config);
        for instance in 0..5 {
            for (left_offset, _, coefficient) in &shared_terms {
                dense_shared = field_config.add(
                    &(dense_shared),
                    &(&(field_config.mul(
                        &(field_config.mul(
                            &(coefficient.clone()),
                            &(&left_equality[7 * instance + left_offset]),
                        )),
                        &(&right_equality[0]),
                    ))),
                );
            }
        }
        assert_eq!(optimized_shared, dense_shared);
    }

    #[test]
    fn non_power_sha_terminal_contractions_match_dense_evaluation() {
        let prepared = prepare_sha256_compression_batch_for_assignment_rows(21)
            .expect("102-instance packed test relation");
        assert_eq!(prepared.instances(), 102);
        let field_config = Fp::<2>::make_cfg(&Uint::from(FQ_MOD)).expect("fixed test field");
        let relation = prepared
            .project_linear_relation(&field_config)
            .expect("project linear relation");
        let constraint_point = (0..flat_constraint_vars(&prepared).unwrap())
            .map(|coordinate| {
                SpartanBitzField::from_with_cfg((coordinate as u64) + 2, &field_config)
            })
            .collect::<Vec<_>>();
        let assignment_point = (0..prepared.assignment_params().row_vars
            + prepared.assignment_params().col_vars)
            .map(|coordinate| {
                SpartanBitzField::from_with_cfg((coordinate as u64) + 41, &field_config)
            })
            .collect::<Vec<_>>();
        let constraint_weights = compact_eq_table(&constraint_point, &field_config).unwrap();
        let assignment_weights = FactoredEqualityWeights::new(
            &assignment_point,
            prepared.assignment_params().row_vars,
            &field_config,
        )
        .unwrap();
        let dense_relation = evaluate_flat_linear_collapse_dense(
            &prepared,
            &relation,
            &constraint_weights,
            &assignment_weights,
        )
        .unwrap();
        let optimized_relation = evaluate_repeated_flat_linear_collapse(
            &prepared,
            &relation,
            &constraint_point,
            &assignment_point,
        )
        .unwrap();
        assert_eq!(optimized_relation, dense_relation);

        let statements = (0..prepared.instances())
            .map(|instance| Sha256CompressionStatement::new(input(instance), [instance as u32; 8]))
            .collect::<Vec<_>>();
        let instance_point =
            public_instance_point(&constraint_point, prepared.instances()).unwrap();
        let slot_weights = (0..SHA256_PUBLIC_WORDS * SHA256_PUBLIC_WORD_BITS)
            .map(|slot| SpartanBitzField::from_with_cfg(slot as u64 + 71, &field_config))
            .collect::<Vec<_>>();
        let public_batch_weight = SpartanBitzField::from_with_cfg(83_u64, &field_config);
        let constant_weight = SpartanBitzField::from_with_cfg(89_u64, &field_config);
        let batching = PublicLinearBatching::new(
            &prepared,
            &statements,
            instance_point,
            slot_weights.clone(),
            public_batch_weight.clone(),
            constant_weight.clone(),
            &field_config,
        )
        .unwrap();
        assert_eq!(
            batching.evaluate(&assignment_point, &field_config).unwrap(),
            batching
                .evaluate_dense(&assignment_weights, &field_config)
                .unwrap()
        );

        let instance_weights = eq_table(instance_point, &field_config).unwrap();
        let mut dense_initial_claim = constant_weight;
        for (instance, statement) in statements.iter().enumerate() {
            for (word_slot, word) in statement.words().enumerate() {
                for bit in 0..SHA256_PUBLIC_WORD_BITS {
                    if word >> bit & 1 == 1 {
                        let slot = word_slot * SHA256_PUBLIC_WORD_BITS + bit;
                        let coefficient = field_config.mul(
                            &(field_config.mul(
                                &(instance_weights[instance].clone()),
                                &(&public_batch_weight),
                            )),
                            &(&slot_weights[slot]),
                        );
                        dense_initial_claim =
                            field_config.add(&(dense_initial_claim), &(&coefficient));
                    }
                }
            }
        }
        assert_eq!(*batching.initial_claim(), dense_initial_claim);
    }

    #[test]
    fn factored_opening_matches_the_dense_scaled_equality() {
        let prepared = super::super::constraints::prepare_sha256_compression_batch_for_test(0)
            .expect("one-instance test relation");
        let field_config = Fp::<2>::make_cfg(&Uint::from(FQ_MOD)).expect("fixed test field");
        let h_layout = prepared.assignment_params();
        let assignment_point = (0..h_layout.row_vars + h_layout.col_vars)
            .map(|coordinate| {
                SpartanBitzField::from_with_cfg((coordinate as u64) + 3, &field_config)
            })
            .collect::<Vec<_>>();
        let dense_equality = eq_table(&assignment_point, &field_config).unwrap();
        let factored =
            FactoredEqualityWeights::new(&assignment_point, h_layout.row_vars, &field_config)
                .unwrap();
        let collapsed = SpartanBitzField::from_with_cfg(17_u64, &field_config);
        let initial_claim = SpartanBitzField::from_with_cfg(31_u64, &field_config);
        let (rows, columns, claimed) = linear_opening_claim(
            &prepared,
            h_layout,
            factored,
            &collapsed,
            initial_claim.clone(),
            &field_config,
        )
        .unwrap();

        for (flat_cell, equality) in dense_equality.iter().enumerate() {
            let row = flat_cell & (h_layout.rows() - 1);
            let column = flat_cell >> h_layout.row_vars;
            let row_weight = field_from_raw(rows[row], &field_config);
            let column_weight = SpartanBitzField::from_with_cfg(columns[column], &field_config);
            assert_eq!(
                field_config.mul(&(row_weight), &(&column_weight)),
                field_config.mul(&(equality.clone()), &(&collapsed))
            );
        }
        assert_eq!(claimed, u128::from(field_config.to_integer(&initial_claim)));
    }
}
