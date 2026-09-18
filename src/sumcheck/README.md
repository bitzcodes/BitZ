# Shared outer sumcheck

The BitZ outer provers live in `src/sumcheck/outer`. They share one transcript
state machine (`engine.rs`) and preserve `SumcheckProof<E, 4>`: four cubic
coefficients are stored, absorbed, and serialized per ordinary round. Inner
sumcheck remains `SumcheckProof<E, 3>`. Existing enclosing Spartan proof types
and wire encodings are unchanged.

## Public API

```rust
use bitz::sumcheck::{outer::*, UngrindedRoundBoundary};
use field::{create_prime_field, Uint};

let field = create_prime_field(Uint::<2>::from((1u128 << 100) - 15));
let inputs = OuterInputs {
    ax: vec![2u64, 3],
    bx: vec![5u64, 7],
    cx: vec![10u128, 21],
};
// tau is sampled by the containing protocol before calling this function.
let output = prove_outer_sumcheck(
    &field, &mut transcript, OuterClaim::RowwiseZero, &tau, inputs, None,
    &mut UngrindedRoundBoundary,
)?;
// output.proof: SumcheckProof<field::Fp<2>, 4>
// output.point, output.final_claim, output.evaluations.{ax,bx,cx}
```

- `prove_outer_sumcheck(field, transcript, claim, tau, rows, factors, boundary)`
  proves the ordinary equality-weighted relation. `OuterClaim::Sum(value)` allows
  arbitrary claims, including zero with nonzero individual residuals.
  `OuterClaim::RowwiseZero` promises `A(x)B(x)=C(x)` at every Boolean row and
  enables the first-round zerocheck optimization.
- `prepare_univariate_skip(field, k)` prepares field-specific interpolation
  constants for `1 <= k <= 4`. `prove_outer_zerocheck_with_skip(field, transcript,
  prepared, tau_tail, rows, factors, boundary)` proves a rowwise-zero prefix
  and an ordinary tail. `k == log2(rows)` produces an empty tail.
- Both provers take any `OuterRows`: `OuterInputs<AB,C>` owns three vectors,
  `OuterSlices<AB,C>` borrows three slices, and split-product/circuit storage
  implements indexed row access. Pass `&rows` to retain owned storage; otherwise
  it is dropped immediately after the first mixed fold. No `_from_slices`,
  native/field forwarding functions, or runtime width-dispatch enum is needed.
- `factors: None` prepares equality tables inside the prover. `Some(factors)`
  reuses fresh tables for the same challenge point and coordinate order.
  `EqualityFactors::from_mles` checks dense-table metadata;
  `EqualityFactors::new` accepts vectors. Provers check domain lengths, but the
  caller must ensure the supplied values encode the requested equality point.
- `verify_outer_sumcheck` checks ordinary proofs in either claim mode;
  `verify_outer_zerocheck_with_skip` also checks the prefix.

A/B share their arithmetic type; C can be wider. `OuterArithmetic<AB,C>` and
`OuterRows` provide static arithmetic and storage capabilities. The provider's
associated `Elem` is the folded representation. The existing `SpartanField`
contract supplies transcript encoding and challenge sampling; `vendor/field`
supplies exact products, reductions, accumulators, and prepared folds.

`OuterOutput<E>` contains the proof, challenge point, final claim, and
`OuterEvaluations<E> { ax, bx, cx }`. `SkippedOuterOutput<E>` contains an
`UnivariateSkipProof<E>`, its challenge, and the ordinary tail output.
Verifier outputs contain the corresponding transcript-derived point/claim.

## Arithmetic and protocol boundaries

`engine::RoundState` owns the current claim, bound equality scale, accumulated
round messages, and challenge point. Only its sampling method advances the
ordinary transcript. Prefixes retain absolute round numbering. The equality
scale is multiplied forward and is never divided out, including when it is zero.
Terminal consistency is checked in release builds for every input representation.

The generic engine preserves two-level equality accumulation, parallel block
sizes, reusable fold scratch, and fused fold/next-message passes. Arithmetic is
selected by the row types at compile time. Split products and public zero padding
are read on demand without constructing a C table. Integer residuals stay exact
through initial coefficient accumulation; challenge folding produces smaller
field tables. Arithmetic differences live in `OuterArithmetic` implementations.

Univariate prefix folding uses **Lagrange** weights. `FoldPairs::fold_prefix_into`
in `vendor/field` instead implements Boolean folding with already-known
challenges:

```text
pair:   dst[i] = (1-r) src[2i] + r src[2i+1]
prefix: dst[i] = sum_b eq(challenges,b) src[2^k*i+b]
```

The first challenge binds the low bit. Pair APIs derive the extent from slice
lengths; mapped APIs derive it from the destination. The default multi-round
helper allocates `2^k` equality weights and reads each source entry once through
batch MACs. It cannot combine interactive rounds whose challenges have not yet
been sampled.

## Layout and cleanup

```text
src/sumcheck/
  mod.rs             shared exports
  proof.rs           SumcheckProof, polynomial verification, composition adapters
  boundary.rs        ordinary and grinding transcript policies
  error.rs           shared errors
  arithmetic.rs      exact accumulator and fold helpers shared with inner kernels
  outer/
    mod.rs
    api.rs           direct ordinary prover with explicit claim mode
    engine.rs        shared round state and terminal check
    ordinary.rs      field arithmetic, two-level equality, fused continuation
    arithmetic.rs    native storage and test-only encoded arithmetic
    arithmetic/native.rs   split-product row storage and test-only legacy kernels
    univariate_api.rs      prepared mixed-input prefix API
    univariate.rs          interpolation, prefix proof, existing composition types
    native_skip.rs         test-only independent native skip reference
    tests.rs
```

The old outer drivers and kernels were removed from
`src/piop/spartan/sumcheck.rs` and `raw_monty.rs`; those files retain the inner
prover. The old `univariate_skip.rs`, `univariate_skip_native.rs`, and
`raw_monty/native_products.rs` files are deleted. `NativeOuterInput` and
`WideSource` are removed. Two unused native PIOP composition alternatives are
also deleted. Historical direct-cubic and generic native arithmetic are retained
under tests as independent differential oracles. Compatibility exports preserve
existing proof import paths. Flock and the general multilinear/GKR sumchecks are
outside this change.

## Validation and performance reference

Baseline: `a3450385` (`Unify BitZ arithmetic under vendor/field`), the former outer-kernel
benchmark, default parallel feature plus `span-metrics`, Apple Silicon.
The frozen pre-refactor executable and logs were retained in `/tmp/bitz-outer-baseline`
and `/tmp/bitz-outer-before.log` during development. The current outer-kernel entry point is below. Compare revisions using the same
toolchain, features, thread count, and Perfetto processor:

```sh
PERFETTO_TRACE_PROCESSOR=/path/to/trace_processor_shell \
  cargo bench --bench mul_bitz --features span-metrics,bench-internals -- \
  outer --workload u32-full --log-n 15,17 --reps 5
```

Correctness checks compare coefficients, points, terminal evaluations, and the
continuing transcript against direct cubic arithmetic and native skip kernels.
Coverage includes ordinary zero weighted claims with nonzero row residuals,
zero/one tau coordinates, vanishing bound equality, zero-variable tables, all
skip widths including an empty tail, signed/four-limb inputs, malformed shapes,
and grinding replay. Production u32/u64/u128 and existing PIOP integration tests
remain in the Spartan suite. x86 timing and allocation/peak-memory qualification
must be run on the corresponding benchmark hosts; no x86 speedup is inferred
from Apple Silicon results.

### Local results (2026-09-15)

Two alternating runs of each binary, 7 measured repetitions after 1 warmup per
case, 10 Rayon threads, identical input seeds, LTO and bench-profile symbols.
All measured proofs verified. The second pair is shown below;
[performance.csv](performance.csv) preserves all four runs. Timing drift is
visible in unchanged inner work as well, so these are local regression
measurements, not a portable speedup estimate. No consistent outer slowdown
appeared across these runs.

| Rows | Variant | Baseline outer (ms) | Shared outer (ms) |
| --- | --- | ---: | ---: |
| 2^15 | standard | 1.089 | 0.904 |
| 2^15 | skip-k1 | 0.938 | 0.920 |
| 2^15 | skip-k2 | 0.831 | 0.727 |
| 2^15 | skip-k3 | 0.595 | 0.591 |
| 2^15 | skip-k4 | 1.027 | 1.032 |
| 2^17 | standard | 2.038 | 1.738 |
| 2^17 | skip-k1 | 1.975 | 1.960 |
| 2^17 | skip-k2 | 1.675 | 1.479 |
| 2^17 | skip-k3 | 1.336 | 1.146 |
| 2^17 | skip-k4 | 1.355 | 1.250 |

Checks completed: 246 Spartan tests passed (2 ignored); 20 shared outer tests
passed; 19 outer tests passed in the serial build before the final grinding
parity test was added; the ECDSA 128-bit profile proof/extra-nonce test passed;
the field crate suite and the added multi-round fold test passed.
`cargo check --lib --features ecdsa,span-metrics` passed. Existing repository
warnings remain. No x86 or peak-memory result is claimed.

### Generic API parallelism

The generic native-to-field first fold uses disjoint Rayon output chunks above
the existing parallel cutoff. Transcript operations remain sequential; the skip
prefix and native production kernels are unchanged. The locally archived optimization report
records latency, allocation, and proof-parity comparisons, including the fusion,
prefix-parallelism, and scratch-reuse prototypes that were not retained.

The stronger locally archived pre-unification comparison
against `a3450385` found substantial regressions when the public generic API
replaces specialized kernels. The generic-before/after gains above do not qualify
it as a performance-equivalent replacement for the production paths.
