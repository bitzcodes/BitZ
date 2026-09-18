//! Reduce the outer matrix claims and linear/public equations to
//! `claimed_sum = Σ_i batched_matrix[i] · witness[i]` over the sampled field.
//! The prover prepares the matrix MLE; the verifier evaluates it directly.

use crate::piop::spartan::SpartanField as _;
use crate::piop::spartan::raw_monty::RawFieldStorage;
use field::RingOps;

#[cfg(test)]
use super::reduce_integer_mod_q;
use super::{
    error,
    relation::{OuterMode, PreparedSha256Ecdsa, SHA_H, Sha256EcdsaStatement},
};
use crate::piop::spartan::{
    matrix::eq_table,
    raw_monty::{raw_to_words, words_to_raw},
    sumcheck::OuterSumcheckProof,
};
use circuit::linear_map::circuit::PreparedWengertEvaluator;

use crate::sumcheck::bridge::composite::{
    CompositeBinding, CompositeCoefficients, CompositeRows, RawEqualityWeights, equality_weights,
};
#[cfg(test)]
use crate::sumcheck::bridge::composite::{evaluate_sha_factors, evaluate_tail_by_runs};

#[cfg(test)]
use field::Uint;

/// Compact description of the equation checked by the inner sumcheck.
pub(super) struct InnerSumcheckClaim {
    /// Final row-evaluation point returned by the outer sumcheck.
    outer_row_point: Vec<field::Fp<2>>,
    /// Challenge `c` combining the A/B/C claims with weights `1, c, c²`.
    matrix_batch_challenge: field::Fp<2>,
    /// Point defining equality weights for the linear and public-input equations.
    linear_row_point: Vec<field::Fp<2>>,
    /// Scales the linear/public batch when adding it to the matrix batch.
    linear_batch_weight: field::Fp<2>,
    /// Claimed `Σ_i batched_matrix[i] · witness[i]` over assignment indices.
    claimed_sum: field::Fp<2>,
}

/// Distinct integer matrix coefficients reduced modulo the sampled prime, and
/// the P-256 circuit's reverse-mode tape prepared for that prime.
pub(super) struct ModQCoefficients<'a> {
    /// Arithmetic modulo the transcript-sampled prime.
    ctx: field::FpCtx<2>,
    /// One Montgomery residue per distinct `LocalRelation::coefficients` entry.
    residues: Vec<u128>,
    /// `r · (A + xB + x²C)` over the P-256 tail through the circuit's DAG. Its
    /// Montgomery form is the same shared two-limb representation as [`u128`].
    tape: PreparedWengertEvaluator<'a>,
}

impl InnerSumcheckClaim {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_outer_claims(
        relation: &PreparedSha256Ecdsa,
        statement: &Sha256EcdsaStatement,
        outer: &OuterSumcheckProof<field::Fp<2>>,
        outer_row_point: Vec<field::Fp<2>>,
        matrix_batch_challenge: field::Fp<2>,
        linear_row_point: Vec<field::Fp<2>>,
        linear_batch_weight: field::Fp<2>,
        cfg: &field::FpCtx<2>,
    ) -> Result<Self, super::Sha256EcdsaError> {
        if outer_row_point.len() != relation.outer_sumcheck_num_vars()
            || linear_row_point.len() != relation.linear_vars()
        {
            return Err(error("inner batching point dimension mismatch"));
        }
        let squared_challenge = cfg.mul(
            &(matrix_batch_challenge.clone()),
            &(&matrix_batch_challenge),
        );
        let mut claimed_sum = outer.az_mle_claim.clone();
        claimed_sum = cfg.add(
            &(claimed_sum),
            &(&(cfg.mul(&(outer.bz_mle_claim.clone()), &(&matrix_batch_challenge)))),
        );
        claimed_sum = cfg.add(
            &(claimed_sum),
            &(&(cfg.mul(&(outer.cz_mle_claim.clone()), &(&squared_challenge)))),
        );
        let linear_weights = equality_weights(&linear_row_point, cfg)?;
        let public_start = public_row_start(relation);
        claimed_sum = cfg.add(
            &(claimed_sum),
            &(&(cfg.mul(&(linear_weights.at(public_start)), &(&linear_batch_weight)))),
        );
        for bit in 0..1024 {
            if statement.bit(bit) {
                claimed_sum = cfg.add(
                    &(claimed_sum),
                    &(&(cfg.mul(
                        &(linear_weights.at(public_start + 1 + bit)),
                        &(&linear_batch_weight),
                    ))),
                );
            }
        }
        Ok(Self {
            outer_row_point,
            matrix_batch_challenge,
            linear_row_point,
            linear_batch_weight,
            claimed_sum,
        })
    }

    pub(super) fn claimed_sum(&self) -> &field::Fp<2> {
        &self.claimed_sum
    }
}

impl<'a> ModQCoefficients<'a> {
    pub(super) fn from_relation(relation: &'a PreparedSha256Ecdsa, cfg: &field::FpCtx<2>) -> Self {
        let _scope = tracing::info_span!("ecdsa:matrix_projection").entered();
        let ctx = crate::piop::spartan::raw_monty::field_context(cfg);
        let projection = field::PreparedSignedProjection::new(
            ctx.clone(),
            relation.local.coefficients.max_limbs(),
        );
        let residues = relation
            .local
            .coefficients
            .iter()
            .map(|coefficient| {
                let value = projection.project(coefficient);
                u128::from(*value.as_montgomery_integer())
            })
            .collect();
        let tape = relation.local.tape.prepare_field(&ctx);
        Self {
            ctx,
            residues,
            tape,
        }
    }

    /// `Σ_row Σ_m weight[3·row + m] · M_m[row]` over every P-256 tail column, by
    /// the reverse-mode tape. Both sides use the shared two-limb Montgomery
    /// form with `R = 2^128`, so the words convert to [`u128`] without arithmetic.
    #[cfg(test)]
    fn tape_tail(
        &mut self,
        relation: &PreparedSha256Ecdsa,
        matrix_rows: &[u128],
    ) -> Result<Vec<field::Fp<2>>, super::Sha256EcdsaError> {
        let mut output = self.ctx.zero_vec(relation.local.tail.column_count());
        self.tape
            .adjoint_map_into(
                matrix_rows.len() / 3,
                |row, kind| raw_to_words(matrix_rows[3 * row + kind]),
                &mut output,
            )
            .map_err(|e| error(format!("P-256 tape: {e}")))?;
        Ok(output)
    }

    /// The geometric runs of the tape's last output, as field bases:
    /// `tail[start + k] = base · 2^k`. With `split_public`, every run is cut
    /// around the public-bit cells (whose values the prover adjusts afterwards).
    #[cfg(test)]
    fn tail_runs(
        &self,
        relation: &PreparedSha256Ecdsa,
        cfg: &field::FpCtx<2>,
        split_public: bool,
    ) -> Vec<(usize, usize, field::Fp<2>)> {
        let two = field::Fp::<2>::from_with_cfg(2u64, cfg);
        let mut exceptions: Vec<usize> = if split_public {
            relation.local.public_h.to_vec()
        } else {
            Vec::new()
        };
        exceptions.sort_unstable();
        let mut out = Vec::new();
        for run in self.tape.power_runs() {
            let mut start = run.first_column;
            let end = run.first_column + run.len;
            let mut base_at_start =
                crate::utils::delayed_reduction::element(&cfg, words_to_raw(&run.base));
            let first = exceptions.partition_point(|&c| c < start);
            for &cell in &exceptions[first..] {
                if cell >= end {
                    break;
                }
                if cell > start {
                    out.push((start, cell - start, base_at_start.clone()));
                }
                // Skip the exceptional cell; the run continues after it with
                // its base advanced by 2^(cell + 1 - start).
                let mut advanced = base_at_start.clone();
                for _ in start..=cell {
                    advanced = cfg.mul(&(advanced), &(&two));
                }
                base_at_start = advanced;
                start = cell + 1;
            }
            if start < end {
                out.push((start, end - start, base_at_start));
            }
        }
        out
    }

    /// Builds the multilinear coefficient polynomial over `F_q`:
    ///
    /// ```text
    /// V(y) = A_o(r, y) + ρ B_o(r, y) + ρ² C_o(r, y) + γ L(s, y).
    /// ```
    ///
    /// Here `r = outer_row_point`, `s = linear_row_point`,
    /// `ρ = matrix_batch_challenge`, and `γ = linear_batch_weight`.
    /// `A_o`, `B_o`, and `C_o` are the multilinear extensions of the coefficient
    /// maps for rows selected by the outer mode. `L` is the corresponding
    /// extension for the left-hand sides of the remaining linear equations,
    /// `h[0] = 1`, and the public-bit equations, using the protocol's row
    /// ordering and zero padding. With `h` the assignment MLE in `n` variables,
    /// the inner sumcheck checks `Σ_{y ∈ {0,1}^n} V(y) h(y) = claimed_sum`.
    pub(super) fn build_batched_matrix_mle(
        &mut self,
        relation: &PreparedSha256Ecdsa,
        claim: &InnerSumcheckClaim,
        cfg: &field::FpCtx<2>,
    ) -> Result<CompositeCoefficients, super::Sha256EcdsaError> {
        let _scope = tracing::info_span!("ecdsa:coefficient_combine").entered();
        let weights = self.build_row_weights(relation, claim);
        let (instances, local) = self.build_sha_factors(relation, claim, cfg)?;
        Ok(CompositeBinding {
            field: &self.ctx,
            tape: &mut self.tape,
            num_vars: relation.h_layout.row_vars + relation.h_layout.col_vars,
            tail_offset: relation.map.h_offset,
            tail_columns: relation.local.tail.column_count(),
        }
        .bind_rows(&CompositeRows {
            instances: &instances,
            local: &local,
            tail_rows: &weights.matrix_rows,
            correction_columns: &relation.local.public_h,
            corrections: &weights.public_bits,
            constant: weights.constant,
        })?)
    }

    /// Evaluate the public matrix MLE at a point: the SHA part by its factored
    /// form, the P-256 tail by a FORWARD pass of the tape — the batched rows
    /// `Σ_m weight[3·row + m] · M_m[row]` evaluated at the tail cells' equality
    /// weights and closed with the row weights — and the public bits on top.
    /// The tail term is the bilinear form `⟨w, M·eq⟩ = ⟨Mᵀw, eq⟩` that the
    /// reverse pass ([`Self::evaluate_batched_matrix_mle_reverse`], the test
    /// oracle) computes as the tape's column vector dotted with the equality
    /// weights; the forward pass never materializes that vector.
    pub(super) fn evaluate_batched_matrix_mle(
        &mut self,
        relation: &PreparedSha256Ecdsa,
        claim: &InnerSumcheckClaim,
        assignment_point: &[field::Fp<2>],
        cfg: &field::FpCtx<2>,
    ) -> Result<field::Fp<2>, super::Sha256EcdsaError> {
        let _scope = tracing::info_span!("ecdsa:coefficient_evaluate").entered();
        check_assignment_point(
            relation.h_layout.row_vars + relation.h_layout.col_vars,
            assignment_point,
        )?;
        let ctx = &self.ctx;
        let weights = {
            let _scope = tracing::info_span!("ecdsa:ce_rows").entered();
            self.build_row_weights(relation, claim)
        };
        let (instances, sha) = {
            let _scope = tracing::info_span!("ecdsa:ce_sha_factors").entered();
            self.build_sha_factors(relation, claim, cfg)?
        };
        Ok(CompositeBinding {
            field: ctx,
            tape: &mut self.tape,
            num_vars: relation.h_layout.row_vars + relation.h_layout.col_vars,
            tail_offset: relation.map.h_offset,
            tail_columns: relation.local.tail.column_count(),
        }
        .evaluate_at(
            &CompositeRows {
                instances: &instances,
                local: &sha,
                tail_rows: &weights.matrix_rows,
                correction_columns: &relation.local.public_h,
                corrections: &weights.public_bits,
                constant: weights.constant,
            },
            assignment_point,
        )?)
    }

    /// The reverse-mode evaluation the forward pass replaced: the tape's
    /// column vector (one reverse pass over the DAG), then the run-structured
    /// weighted sum over it. Kept as the test oracle.
    #[cfg(test)]
    pub(super) fn evaluate_batched_matrix_mle_reverse(
        &mut self,
        relation: &PreparedSha256Ecdsa,
        claim: &InnerSumcheckClaim,
        assignment_point: &[field::Fp<2>],
        cfg: &field::FpCtx<2>,
    ) -> Result<field::Fp<2>, super::Sha256EcdsaError> {
        check_assignment_point(
            relation.h_layout.row_vars + relation.h_layout.col_vars,
            assignment_point,
        )?;
        let weights = self.build_row_weights_field(relation, claim, cfg)?;
        let tail = self.tape_tail(relation, &weights.matrix_rows)?;
        // The verifier adds the public-bit weights separately below, so the
        // tape's runs describe `tail` unsplit.
        let runs = self.tail_runs(relation, cfg, false);
        let (instances, sha) = self.build_sha_factors_field(relation, claim, cfg)?;
        let equality = equality_weights(assignment_point, cfg)?;
        let mut value = evaluate_sha_factors(&instances, &sha, assignment_point, cfg)?;
        value = cfg.add(
            &(value),
            &(&(cfg.mul(&(weights.constant), &(&equality.at(0))))),
        );
        value = cfg.add(
            &(value),
            &(&evaluate_tail_by_runs(
                cfg,
                &self.ctx,
                relation.map.h_offset,
                &tail,
                &runs,
                &equality,
            )),
        );
        for (&cell, &weight) in relation.local.public_h.iter().zip(&weights.public_bits) {
            value = cfg.add(
                &(value),
                &(&(cfg.mul(
                    &(crate::utils::delayed_reduction::element(&cfg, weight)),
                    &(&equality.at(relation.map.h_offset + cell)),
                ))),
            );
        }
        Ok(value)
    }

    /// Computes the P-256 row and public-equation weights over `F_q`.
    /// Let `e_o(i) = eq(outer_row_point, i)`, `e_l(i) = eq(linear_row_point, i)`,
    /// `ρ = matrix_batch_challenge`, `γ = linear_batch_weight`, and
    /// `S = 256 · compressions()`, with integer indices interpreted as Bit
    /// vectors. For `w(i) = (w_A(i), w_B(i), w_C(i))`, the matrix weights are:
    ///
    /// ```text
    /// Split:   w(nonlinear[k]) = e_o(k) · (1, ρ, ρ²)
    ///          w(linear[k])    = (0, 0, γ e_l(S + k))
    /// AllRows: w(i)            = e_o(S + i) · (1, ρ, ρ²).
    /// ```
    ///
    /// With `P = S + |linear|`, the remaining weights are
    /// `constant = γ e_l(P)` and `public_bits[b] = γ e_l(P + 1 + b)`
    /// for `0 ≤ b < 1024`.
    fn build_row_weights(
        &self,
        relation: &PreparedSha256Ecdsa,
        claim: &InnerSumcheckClaim,
    ) -> RowWeights {
        let ctx = &self.ctx;
        let linear = RawEqualityWeights::new(ctx, &claim.linear_row_point);
        let outer = RawEqualityWeights::new(ctx, &claim.outer_row_point);
        let batch = ctx.raw(&claim.matrix_batch_challenge);
        let batch_squared = ctx.mul_raw(batch, batch);
        let linear_batch = ctx.raw(&claim.linear_batch_weight);
        let mut matrix_rows = vec![0 as u128; 3 * relation.local.rows()];
        let matrix_weights = |slots: &mut [u128], weight| {
            slots[0] = weight;
            slots[1] = ctx.mul_raw(weight, batch);
            slots[2] = ctx.mul_raw(weight, batch_squared);
        };
        match relation.mode {
            OuterMode::Split => {
                for (index, &row) in relation.local.nonlinear.iter().enumerate() {
                    matrix_weights(&mut matrix_rows[3 * row..3 * row + 3], outer.at(index));
                }
                for (index, &row) in relation.local.linear.iter().enumerate() {
                    matrix_rows[3 * row + 2] = ctx.mul_raw(
                        linear.at(256 * relation.compressions() + index),
                        linear_batch,
                    );
                }
            }
            OuterMode::AllRows => {
                for row in 0..relation.local.rows() {
                    let weight = outer.at(256 * relation.compressions() + row);
                    matrix_weights(&mut matrix_rows[3 * row..3 * row + 3], weight);
                }
            }
        }
        let public_start = public_row_start(relation);
        let constant = ctx.mul_raw(linear.at(public_start), linear_batch);
        let public_bits = (0..1024)
            .map(|bit| ctx.mul_raw(linear.at(public_start + 1 + bit), linear_batch))
            .collect();
        RowWeights {
            matrix_rows,
            public_bits,
            constant,
        }
    }

    /// [`Self::build_row_weights`] with the equality weights and the products
    /// taken in the field domain; kept as the test oracle.
    #[cfg(test)]
    fn build_row_weights_field(
        &self,
        relation: &PreparedSha256Ecdsa,
        claim: &InnerSumcheckClaim,
        cfg: &field::FpCtx<2>,
    ) -> Result<RowWeightsField, super::Sha256EcdsaError> {
        let ctx = &self.ctx;
        let linear = equality_weights(&claim.linear_row_point, cfg)?;
        let outer = equality_weights(&claim.outer_row_point, cfg)?;
        let batch = ctx.raw(&claim.matrix_batch_challenge);
        let batch_squared = ctx.mul_raw(batch, batch);
        let mut matrix_rows = vec![0 as u128; 3 * relation.local.rows()];
        let matrix_weights = |slots: &mut [u128], weight| {
            slots[0] = weight;
            slots[1] = ctx.mul_raw(weight, batch);
            slots[2] = ctx.mul_raw(weight, batch_squared);
        };
        match relation.mode {
            OuterMode::Split => {
                for (index, &row) in relation.local.nonlinear.iter().enumerate() {
                    matrix_weights(
                        &mut matrix_rows[3 * row..3 * row + 3],
                        ctx.raw(&outer.at(index)),
                    );
                }
                for (index, &row) in relation.local.linear.iter().enumerate() {
                    let weight = cfg.mul(
                        &(linear.at(256 * relation.compressions() + index)),
                        &(&claim.linear_batch_weight),
                    );
                    matrix_rows[3 * row + 2] = ctx.raw(&weight);
                }
            }
            OuterMode::AllRows => {
                for row in 0..relation.local.rows() {
                    let weight = ctx.raw(&outer.at(256 * relation.compressions() + row));
                    matrix_weights(&mut matrix_rows[3 * row..3 * row + 3], weight);
                }
            }
        }
        let public_start = public_row_start(relation);
        let constant = cfg.mul(&(linear.at(public_start)), &(&claim.linear_batch_weight));
        let public_bits = (0..1024)
            .map(|bit| {
                ctx.raw(
                    &(cfg.mul(
                        &(linear.at(public_start + 1 + bit)),
                        &(&claim.linear_batch_weight),
                    )),
                )
            })
            .collect();
        Ok(RowWeightsField {
            matrix_rows,
            public_bits,
            constant,
        })
    }

    /// The arbitrary-precision projection of the distinct coefficients the
    /// native word reduction replaced; kept as the test oracle.
    #[cfg(test)]
    fn bigint_residues(
        relation: &PreparedSha256Ecdsa,
        modulus: u128,
        cfg: &field::FpCtx<2>,
    ) -> Vec<u128> {
        let ctx = crate::piop::spartan::raw_monty::field_context(cfg);
        relation
            .local
            .coefficients
            .iter()
            .map(|coefficient| {
                let bytes: Vec<_> = coefficient
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                let integer = num_bigint::BigInt::from_signed_bytes_le(&bytes);
                ctx.raw(&reduce_integer_mod_q(&integer, modulus, cfg))
            })
            .collect()
    }

    /// The expanded-entry gather the tape replaced; kept as the test oracle.
    #[cfg(test)]
    fn p256_column_weight(
        &self,
        relation: &PreparedSha256Ecdsa,
        row_weights: &[u128],
        column: usize,
    ) -> u128 {
        relation
            .local
            .tail
            .column(column)
            .unwrap()
            .indexed_entries()
            .fold(0 as u128, |sum, (slot, coefficient)| {
                self.ctx.add_raw(
                    sum,
                    self.ctx
                        .mul_raw(row_weights[slot], self.residues[coefficient]),
                )
            })
    }

    /// The SHA part's factors `(instances, sha)` in raw residues: the
    /// batched matrix MLE's entry `i + N·j` is `instances[i] · sha[j]`.
    /// `build_sha_factors_field` is the test oracle.
    fn build_sha_factors(
        &self,
        relation: &PreparedSha256Ecdsa,
        claim: &InnerSumcheckClaim,
        cfg: &field::FpCtx<2>,
    ) -> Result<(Vec<u128>, Vec<u128>), super::Sha256EcdsaError> {
        let ctx = &self.ctx;
        let batch = ctx.raw(&claim.matrix_batch_challenge);
        let (point, multiplier) = match relation.mode {
            OuterMode::Split => (&claim.linear_row_point, ctx.raw(&claim.linear_batch_weight)),
            OuterMode::AllRows => (&claim.outer_row_point, ctx.mul_raw(batch, batch)),
        };
        let instances = ctx.raw_vec(&eq_table(&point[..relation.log_n], cfg).map_err(error)?);
        let local_weights = RawEqualityWeights::new(ctx, &point[relation.log_n..]);
        let mut sha = vec![0 as u128; SHA_H];
        for row in 0..relation.local.sha_c.row_count() {
            let weight = ctx.mul_raw(local_weights.at(row), multiplier);
            for (column, coefficient) in relation.local.sha_c.row(row).unwrap().indexed_entries() {
                sha[column] =
                    ctx.add_raw(sha[column], ctx.mul_raw(weight, self.residues[coefficient]));
            }
        }
        Ok((instances, sha))
    }

    /// [`Self::build_sha_factors`] in the field domain; kept as the test oracle.
    #[cfg(test)]
    fn build_sha_factors_field(
        &self,
        relation: &PreparedSha256Ecdsa,
        claim: &InnerSumcheckClaim,
        cfg: &field::FpCtx<2>,
    ) -> Result<(Vec<field::Fp<2>>, Vec<field::Fp<2>>), super::Sha256EcdsaError> {
        let squared_challenge = cfg.mul(
            &(claim.matrix_batch_challenge.clone()),
            &(&claim.matrix_batch_challenge),
        );
        let (point, multiplier) = match relation.mode {
            OuterMode::Split => (&claim.linear_row_point, &claim.linear_batch_weight),
            OuterMode::AllRows => (&claim.outer_row_point, &squared_challenge),
        };
        let instances = eq_table(&point[..relation.log_n], cfg).map_err(error)?;
        let local_weights = equality_weights(&point[relation.log_n..], cfg)?;
        let multiplier = self.ctx.raw(multiplier);
        let mut sha = vec![0 as u128; SHA_H];
        for row in 0..relation.local.sha_c.row_count() {
            let weight = self
                .ctx
                .mul_raw(self.ctx.raw(&local_weights.at(row)), multiplier);
            for (column, coefficient) in relation.local.sha_c.row(row).unwrap().indexed_entries() {
                sha[column] = self.ctx.add_raw(
                    sha[column],
                    self.ctx.mul_raw(weight, self.residues[coefficient]),
                );
            }
        }
        Ok((
            instances,
            sha.into_iter()
                .map(|value| crate::utils::delayed_reduction::element(&cfg, value))
                .collect(),
        ))
    }
}

struct RowWeights {
    /// Slot `3 * row + matrix` for A/B/C.
    matrix_rows: Vec<u128>,
    public_bits: Vec<u128>,
    constant: u128,
}

/// The field-domain oracle's form of [`RowWeights`].
#[cfg(test)]
struct RowWeightsField {
    matrix_rows: Vec<u128>,
    public_bits: Vec<u128>,
    constant: field::Fp<2>,
}

fn public_row_start(relation: &PreparedSha256Ecdsa) -> usize {
    256 * relation.compressions() + relation.local.linear.len()
}

fn check_assignment_point(
    num_vars: usize,
    point: &[field::Fp<2>],
) -> Result<(), super::Sha256EcdsaError> {
    if point.len() != num_vars {
        return Err(error("assignment point dimension mismatch"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piop::spartan::{
        ecdsa_sha256::{prepare_sha256_ecdsa, tests::fixture},
        sumcheck::SumcheckProof,
    };

    /// A 113-bit prime drawn the way the protocol draws it (from a fresh
    /// transcript): the modulus class the verifier runs on, next to the wider
    /// Mersenne prime the other oracle tests use.
    pub(super) fn sampled_prime() -> u128 {
        crate::ext_proj::sample_prime_in_interval(
            &mut crate::transcript::Blake3Transcript::new(),
            1u128 << 112,
            (1u128 << 113) - 1,
        )
        .unwrap()
    }

    /// The native word reduction of the distinct coefficients equals the
    /// arbitrary-precision projection residue for residue, at a sampled
    /// 113-bit prime and at `2^127 − 1`.
    #[test]
    fn coefficient_residues_match_bigint_projection() {
        for modulus in [sampled_prime(), (1u128 << 127) - 1] {
            let cfg = field::Fp::<2>::make_cfg(&Uint::from(modulus)).unwrap();
            for mode in [OuterMode::Split, OuterMode::AllRows] {
                let relation = prepare_sha256_ecdsa(3, 100, mode).unwrap();
                let coefficients = ModQCoefficients::from_relation(&relation, &cfg);
                assert_eq!(
                    coefficients.residues,
                    ModQCoefficients::bigint_residues(&relation, modulus, &cfg),
                    "{mode:?} modulus {modulus}"
                );
            }
        }
    }

    /// The tape's tail must equal the expanded-entry gather element by element,
    /// for both outer modes (Split weights only the linear rows' `C` slot).
    #[test]
    fn tape_tail_matches_column_gather() {
        let modulus = (1u128 << 127) - 1; // the Mersenne prime M127: two limbs, odd, wider than the sampled primes
        let cfg = field::Fp::<2>::make_cfg(&Uint::from(modulus)).unwrap();
        let f = |n| field::Fp::<2>::from_with_cfg(n, &cfg);
        let (statement, _) = fixture();
        for mode in [OuterMode::Split, OuterMode::AllRows] {
            let relation = prepare_sha256_ecdsa(3, 100, mode).unwrap();
            let outer = OuterSumcheckProof {
                sumcheck: SumcheckProof {
                    round_polynomials: Vec::new(),
                },
                az_mle_claim: f(5u64),
                bz_mle_claim: f(7),
                cz_mle_claim: f(11),
            };
            let claim = InnerSumcheckClaim::from_outer_claims(
                &relation,
                &statement,
                &outer,
                (0..relation.outer_sumcheck_num_vars())
                    .map(|i| f(i as u64 + 13))
                    .collect(),
                f(17),
                (0..relation.linear_vars())
                    .map(|i| f(i as u64 + 19))
                    .collect(),
                f(23),
                &cfg,
            )
            .unwrap();
            let mut coefficients = ModQCoefficients::from_relation(&relation, &cfg);
            let weights = coefficients.build_row_weights(&relation, &claim);
            let tape = coefficients
                .tape_tail(&relation, &weights.matrix_rows)
                .unwrap();
            assert_eq!(tape.len(), relation.local.tail.column_count());
            let gather: Vec<u128> = (0..relation.local.tail.column_count())
                .map(|column| {
                    coefficients.p256_column_weight(&relation, &weights.matrix_rows, column)
                })
                .collect();
            let first_mismatch = tape
                .iter()
                .zip(&gather)
                .position(|(a, b)| coefficients.ctx.raw(a) != *b);
            assert_eq!(first_mismatch, None, "{mode:?}");
        }
    }

    /// Compare compact emission with the independent expanded matrix, including
    /// public-cell corrections in both outer modes and two field contexts.
    #[test]
    fn compact_tail_matches_expanded_matrix() {
        for modulus in [sampled_prime(), (1u128 << 127) - 1] {
            let cfg = field::Fp::<2>::make_cfg(&Uint::from(modulus)).unwrap();
            let f = |n| field::Fp::<2>::from_with_cfg(n, &cfg);
            let (statement, _) = fixture();
            for mode in [OuterMode::Split, OuterMode::AllRows] {
                let relation = prepare_sha256_ecdsa(3, 100, mode).unwrap();
                let outer = OuterSumcheckProof {
                    sumcheck: SumcheckProof {
                        round_polynomials: Vec::new(),
                    },
                    az_mle_claim: f(5u64),
                    bz_mle_claim: f(7),
                    cz_mle_claim: f(11),
                };
                let claim = InnerSumcheckClaim::from_outer_claims(
                    &relation,
                    &statement,
                    &outer,
                    (0..relation.outer_sumcheck_num_vars())
                        .map(|i| f(i as u64 + 13))
                        .collect(),
                    f(17),
                    (0..relation.linear_vars())
                        .map(|i| f(i as u64 + 19))
                        .collect(),
                    f(23),
                    &cfg,
                )
                .unwrap();
                let mut coefficients = ModQCoefficients::from_relation(&relation, &cfg);
                let prepared = coefficients
                    .build_batched_matrix_mle(&relation, &claim, &cfg)
                    .unwrap();
                let weights = coefficients.build_row_weights(&relation, &claim);
                let mut oracle: Vec<u128> = (0..relation.local.tail.column_count())
                    .map(|column| {
                        coefficients.p256_column_weight(&relation, &weights.matrix_rows, column)
                    })
                    .collect();
                for (&cell, &weight) in relation.local.public_h.iter().zip(&weights.public_bits) {
                    oracle[cell] = coefficients.ctx.add_raw(oracle[cell], weight);
                }
                let mut actual = vec![0; prepared.p256_tail.len()];
                prepared
                    .p256_tail
                    .visit(0..actual.len(), |start, len, base| {
                        let mut value = words_to_raw(&base);
                        for entry in &mut actual[start..start + len] {
                            *entry = value;
                            value = coefficients.ctx.add_raw(value, value);
                        }
                    });
                assert_eq!(
                    actual.iter().zip(&oracle).position(|(a, b)| a != b),
                    None,
                    "{mode:?}, {modulus}"
                );
            }
        }
    }

    /// The raw-residue row weights and SHA factors are the field-domain ones
    /// residue for residue (both outer modes, random claims, a sampled 113-bit
    /// prime and `2^127 − 1`).
    #[test]
    fn raw_factor_builders_match_field_builders() {
        let (statement, _) = fixture();
        for modulus in [sampled_prime(), (1u128 << 127) - 1] {
            let cfg = field::Fp::<2>::make_cfg(&Uint::from(modulus)).unwrap();
            let mut state = 0x1357_9BDF_2468_ACE0_u64 ^ modulus as u64;
            let mut random = move || {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            };
            let mut element = || {
                let value = (u128::from(random()) << 64) | u128::from(random());
                field::Fp::<2>::from_with_cfg(value % modulus, &cfg)
            };
            for mode in [OuterMode::Split, OuterMode::AllRows] {
                let relation = prepare_sha256_ecdsa(3, 100, mode).unwrap();
                let outer = OuterSumcheckProof {
                    sumcheck: SumcheckProof {
                        round_polynomials: Vec::new(),
                    },
                    az_mle_claim: element(),
                    bz_mle_claim: element(),
                    cz_mle_claim: element(),
                };
                let claim = InnerSumcheckClaim::from_outer_claims(
                    &relation,
                    &statement,
                    &outer,
                    (0..relation.outer_sumcheck_num_vars())
                        .map(|_| element())
                        .collect(),
                    element(),
                    (0..relation.linear_vars()).map(|_| element()).collect(),
                    element(),
                    &cfg,
                )
                .unwrap();
                let coefficients = ModQCoefficients::from_relation(&relation, &cfg);
                let ctx = &coefficients.ctx;
                let raw = coefficients.build_row_weights(&relation, &claim);
                let field = coefficients
                    .build_row_weights_field(&relation, &claim, &cfg)
                    .unwrap();
                assert_eq!(raw.matrix_rows, field.matrix_rows, "{mode:?} rows");
                assert_eq!(raw.public_bits, field.public_bits, "{mode:?} public bits");
                assert_eq!(
                    crate::utils::delayed_reduction::element(ctx, raw.constant),
                    field.constant,
                    "{mode:?} constant"
                );
                let (instances, sha) = coefficients
                    .build_sha_factors(&relation, &claim, &cfg)
                    .unwrap();
                let (instances_field, sha_field) = coefficients
                    .build_sha_factors_field(&relation, &claim, &cfg)
                    .unwrap();
                assert_eq!(
                    ctx.raw_vec(&instances_field),
                    instances,
                    "{mode:?} instances"
                );
                assert_eq!(ctx.raw_vec(&sha_field), sha, "{mode:?} sha");
                // The grouped raw dot is the factored MLE's evaluation.
                let vars = relation.h_layout.row_vars + relation.h_layout.col_vars;
                let point: Vec<field::Fp<2>> = (0..vars).map(|_| element()).collect();
                let expected =
                    evaluate_sha_factors(&instances_field, &sha_field, &point, &cfg).unwrap();
                let eq_instances = ctx.raw_vec(&eq_table(&point[..relation.log_n], &cfg).unwrap());
                let dot_instances = instances
                    .iter()
                    .zip(&eq_instances)
                    .fold(0 as u128, |sum, (&v, &w)| {
                        ctx.add_raw(sum, ctx.mul_raw(v, w))
                    });
                let local = RawEqualityWeights::new(ctx, &point[relation.log_n..]);
                assert_eq!(
                    crate::utils::delayed_reduction::element(
                        ctx,
                        ctx.mul_raw(dot_instances, local.dot(&sha))
                    ),
                    expected,
                    "{mode:?} sha dot"
                );
            }
        }
    }

    /// The forward-pass evaluation is the reverse-pass evaluation (the
    /// tape's column vector dotted with the equality weights), for both outer
    /// modes, at random claims and points, at a sampled 113-bit prime and at
    /// `2^127 − 1`.
    #[test]
    fn forward_matrix_evaluation_matches_reverse() {
        let (statement, _) = fixture();
        for modulus in [sampled_prime(), (1u128 << 127) - 1] {
            let cfg = field::Fp::<2>::make_cfg(&Uint::from(modulus)).unwrap();
            let mut state = 0x5DEE_CE66_D1B4_E8A3_u64 ^ modulus as u64;
            let mut random = move || {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            };
            let mut element = || {
                let value = (u128::from(random()) << 64) | u128::from(random());
                field::Fp::<2>::from_with_cfg(value % modulus, &cfg)
            };
            for mode in [OuterMode::Split, OuterMode::AllRows] {
                let relation = prepare_sha256_ecdsa(3, 100, mode).unwrap();
                for trial in 0..3 {
                    let outer = OuterSumcheckProof {
                        sumcheck: SumcheckProof {
                            round_polynomials: Vec::new(),
                        },
                        az_mle_claim: element(),
                        bz_mle_claim: element(),
                        cz_mle_claim: element(),
                    };
                    let claim = InnerSumcheckClaim::from_outer_claims(
                        &relation,
                        &statement,
                        &outer,
                        (0..relation.outer_sumcheck_num_vars())
                            .map(|_| element())
                            .collect(),
                        element(),
                        (0..relation.linear_vars()).map(|_| element()).collect(),
                        element(),
                        &cfg,
                    )
                    .unwrap();
                    let vars = relation.h_layout.row_vars + relation.h_layout.col_vars;
                    let point: Vec<field::Fp<2>> = (0..vars).map(|_| element()).collect();
                    let mut coefficients = ModQCoefficients::from_relation(&relation, &cfg);
                    let forward = coefficients
                        .evaluate_batched_matrix_mle(&relation, &claim, &point, &cfg)
                        .unwrap();
                    let reverse = coefficients
                        .evaluate_batched_matrix_mle_reverse(&relation, &claim, &point, &cfg)
                        .unwrap();
                    assert_eq!(forward, reverse, "{mode:?} modulus {modulus} trial {trial}");
                }
            }
        }
    }

    #[test]
    fn streamed_matrix_evaluation_matches_prepared_mle() {
        // The verifier's modulus class (the native word kernels need q > 2^64).
        let modulus = sampled_prime();
        let cfg = field::Fp::<2>::make_cfg(&Uint::from(modulus)).unwrap();
        let f = |n| field::Fp::<2>::from_with_cfg(n, &cfg);
        let (statement, _) = fixture();
        for mode in [OuterMode::Split, OuterMode::AllRows] {
            let relation = prepare_sha256_ecdsa(3, 100, mode).unwrap();
            let outer = OuterSumcheckProof {
                sumcheck: SumcheckProof {
                    round_polynomials: Vec::new(),
                },
                az_mle_claim: f(5u64),
                bz_mle_claim: f(7),
                cz_mle_claim: f(11),
            };
            let claim = InnerSumcheckClaim::from_outer_claims(
                &relation,
                &statement,
                &outer,
                (0..relation.outer_sumcheck_num_vars())
                    .map(|i| f(i as u64 + 13))
                    .collect(),
                f(17),
                (0..relation.linear_vars())
                    .map(|i| f(i as u64 + 19))
                    .collect(),
                f(23),
                &cfg,
            )
            .unwrap();
            let mut coefficients = ModQCoefficients::from_relation(&relation, &cfg);
            let prepared = coefficients
                .build_batched_matrix_mle(&relation, &claim, &cfg)
                .unwrap();
            let mle = prepared.as_mle(&cfg).unwrap();
            let point: Vec<_> = (0..mle.num_vars()).map(|i| f(i as u64 + 29)).collect();
            let cached = prepared.evaluate(&point, &cfg).unwrap();
            assert_eq!(
                coefficients
                    .evaluate_batched_matrix_mle(&relation, &claim, &point, &cfg)
                    .unwrap(),
                cached
            );
            assert_eq!(mle.evaluate(&point, &cfg).unwrap(), cached);
            let tail_start = relation.map.h_offset;
            let tail_end = tail_start + relation.local.tail.column_count();
            for index in [
                0,
                tail_start - 1,
                tail_start,
                tail_end - 1,
                tail_end,
                (1 << mle.num_vars()) - 1,
            ] {
                let point: Vec<_> = (0..mle.num_vars())
                    .map(|bit| f(((index >> bit) & 1) as u64))
                    .collect();
                let expected = mle.evaluation_at(index).unwrap();
                assert_eq!(
                    prepared.evaluate(&point, &cfg).unwrap(),
                    expected,
                    "{mode:?} index={index}"
                );
                assert_eq!(
                    coefficients
                        .evaluate_batched_matrix_mle(&relation, &claim, &point, &cfg)
                        .unwrap(),
                    expected,
                    "{mode:?} index={index}"
                );
            }
            assert!(prepared.evaluate(&point[..point.len() - 1], &cfg).is_err());
            assert!(
                coefficients
                    .evaluate_batched_matrix_mle(&relation, &claim, &point[..point.len() - 1], &cfg)
                    .is_err()
            );
        }
    }
}
