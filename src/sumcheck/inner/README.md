# Inner sumcheck

There are two production prover entry points: `prove_inner_sumcheck` and
`prove_batched_inner_sumcheck`. The single-claim function delegates to the batch
function with one input. The batch function owns the only quadratic proving loop:
reconstructing `q(X) = c0 + (claim - 2*c0 - c2)*X + c2*X²`, absorbing every claim's
message before the shared challenge, applying the round boundary, folding, and
checking terminal products.

## Calls and outputs

```rust
let single = prove_inner_sumcheck(
    field, transcript, claim, values, weights, boundary,
)?;
let batch = prove_batched_inner_sumcheck(
    field, transcript, &claims, values, weights, boundary,
)?;
```

For ordinary inputs, `values` is `Vec<T>` and `weights` is `Vec<F::Elem>`.
An array batch `[Vec<T>; K]` returns `BatchedInnerSumcheckOutput<F::Elem, K>`;
a runtime batch `Vec<Vec<T>>` returns `DynamicBatchedInnerSumcheckOutput<F::Elem>`.
The batch function's generic parameters are inferred as `<F, B>`; `K` comes from
its array input. Claims can be an array reference or slice. No random linear
combination of batch claims is introduced.

The single output contains `proof`, `point`, `final_claim`, and
`terminal_evaluations: [weight, value]`. Batch outputs contain `proofs`, one common
`point`, `final_claims`, and one terminal pair per claim. `SumcheckProof<E, 3>`
stores three coefficients: its degree is two.

Input lengths must agree and be nonempty powers of two. Empty batches and unequal
domains are rejected. Singleton inputs have no rounds and still check their
terminal identity. Invalid public shape, claims metadata, or boundary policy is
rejected before transcript mutation.

## Storage and arithmetic

Sealed input conversions in `input.rs` select preparation, coefficient, fold,
and terminal operations. Callers do not implement a new trait to pass vectors.
The conversions and codecs use static dispatch; the small per-round state enums
never dispatch inside a per-element arithmetic loop.

- `dense`: generic vectors of field elements or integers, with fused conversion,
  folding and next-round accumulation.
- `native`: borrowed native limbs, canonical integer folded storage, live
  prefixes, and block-selector coefficients. `NativeWeights` selects dense or
  block storage. Both can participate in the same batch.
- `packed`: packed bits with factored/composite coefficients and prepared prefixes
  of ordinary rounds. `PackedInput` retains both sources; it uses `()` for the
  separate weights argument. Prefix length does not change the proof.
- `binary`: post-GKR pairs with claims computed in the first pass, selected by
  `InitialClaims::Compute`. Its codec retains the historical evaluation messages,
  header, interpolation nodes and challenge reabsorption. `encode` only converts
  the shared output to the enclosing protocol's existing proof envelope.
- Hybrid packed-bit inputs retain the seven-round lookup prefix, compressed
  message codec and reusable dense scratch buffers. Protocol setup binds the
  random combination before calling the generic prover.

None of these integrations owns a proving loop. The former raw, structured-raw,
SHA-specific and binary batch prover functions, and `engine::drive`, are removed.
Arithmetic kernels remain where their storage-specific optimizations belong.

Arithmetic bounds use `FieldOps`, `BatchMulAcc`, `Reduce`, and
`PreparedLinearCombination` from `vendor/field`, plus `SpartanField` for the
ordinary transcript codec. Streaming MAC is `BatchMulAcc::mul_acc`;
`Reduce::prepare_reduce` selects a schedule from the public total product bound,
including merged workers. Mixed integer differences avoid source-width overflow.
No `SumcheckProductReducer`, `InnerArithmetic`, or `InnerRows` contract is required.

## Verification and qualification

`SumcheckProof::verify_with_round_boundary` and
`SumcheckProof::verify_batch_with_round_boundary` verify ordinary coefficient
messages. Binary and hybrid protocols retain their existing wire-format verifiers.
The enclosing protocol must still discharge terminal matrix and PCS claims.
Legacy algorithms under `cfg(test)` remain independent transcript oracles;
higher-degree proofs retain their existing verifier.

The previous Apple Silicon campaign in `bench_results/inner-refactor-apple`
qualifies the earlier revision. It does not qualify this entry-point removal.
Fresh results are recorded separately in `bench_results/inner-entry-unification-apple`.
The generic fully-field vector first pass still uses the mixed-input three-product
formula; production specialized field kernels retain their two-product formula.

`skipped_experiment` is test-only. Production prefixes prepare ordinary messages;
they do not introduce a univariate-skip proof or change the opening claim.

Current entry-point removal validation: 491 release unit tests, 69 focused
inner/hybrid tests, and 21 release transcript pins passed. The performance
campaign stopped at the requested commit checkpoint after 13/24 cases.
One-thread SHA/P-256 split at 128-bit security measured +1.11% inner time;
MultiSwap batches 1 and 2 measured +2.20% and +1.36%. Their intervals remain
inconclusive against the 1% gate, and follow-up measurements are outstanding.
This revision is not yet fully performance-qualified.
