# Integration status — 2026-09-15

BitZ, circuit, and local Flock now consume `vendor/field` for production arithmetic.
The production graph with `ecdsa` contains none of `crypto-primitives`, its macros,
`crypto-bigint`, `crypto-primes`, `num-bigint`, or `num-integer`.
`vendor/crypto-primitives` and its archived test dependency are removed.
The maintained arithmetic implementation and its tests live in `vendor/field`.

**Performance acceptance is unfinished.** No integrated case has passed the
required median/P95 1% qualification gate. Correctness checks and compile checks
must not be presented as evidence of unchanged performance.

## Baseline and scope

The approved base is `fast-arithmetic` at
`6271724d75570513aa1963381fbf5989933e5b42`. Generated validation logs, benchmark
results, and source archives are not tracked. Recreate measurement artifacts
locally when collecting fresh evidence against that revision.
Root/experiment fat LTO and standalone Flock thin LTO are unchanged.
Limber, Binius64, Plonky3, Spartan2 and ZKPassport arithmetic is outside this scope.

## Arithmetic and consumers

- `Uint<L>` and `Z<L>` retain compile-time limb widths, separate checked/wrapping
  operations, exact products and bounded accumulators. Coefficients and products
  use declared-width segments; private magnitudes do not choose allocation sizes.
- Static prime declarations retain the original moduli. Runtime prime values are
  `Fp<L>` with an explicitly owned `FpCtx<L>`. Elements have no context pointer.
  Runtime modulus association is the caller's contract. Static identities and
  Montgomery product scales remain distinct types.
- General modular arithmetic, prepared division, signed/native mixed MAC,
  canonical-output folds and fixed-schedule private inversion live in `field`.
  Allocation-free `_into` methods and allocating convenience methods coexist.
- BitZ and Flock use the same `Gf8`/`Gf128`; BitZ's `B127` and wide binary
  polynomials also use shared types. GHASH order and AES embeddings are preserved.
  Numeric embedding into characteristic two is parity, separately from explicit
  polynomial-bit construction.
- ARM PMULL/NEON, x86 PCLMUL and four-lane VPCLMUL kernels, prepared multipliers,
  fused rounds, tails and private architecture accumulators are retained in the
  shared crate. Every x86 feature required by a kernel is gated. The four-lane
  packed projection is retained as `PreparedGf128Projection`; obsolete uncalled
  projection helpers were deleted from BitZ, with original source archived.
- u32 keeps borrowed native data. u64/u128 consume native operands and exact
  split products through mixed first-round kernels. u128 x/y/z segments remain
  borrowed. First folds write the representation needed by the next stage.
- Wengert keeps canonical × prepared Montgomery → canonical evaluation,
  Montgomery evaluation, weighted A/B/C reuse and geometric runs. SHA/ECDSA keep
  packed reads, prefix factoring, structured tails and both outer modes.
- Circuit coefficients, preparation, witness hints, P-256 host operations and
  satisfaction checks use fixed-width shared integers. MultiSwap's generator,
  public coefficients and witness checks also use declared-width integers.
- MultiSwap retains its specialized 192-bit low-`u128`/carry-word accumulation
  **at the user's request**. Its exact multiplication and reduction use shared
  arithmetic. Production still prepares reusable MultiSwap residues once; the
  fused alternative exists but has not passed a comparative performance gate.

There is one production delayed Barrett path and no strategy enum. Ordinary
Montgomery multiplication and binary reduction remain. RuntimePrime,
Sha256ModQContext and MultiswapFingerprintContext wrappers are deleted; protocol
profiles remain with prepared relations and samplers return shared contexts.

Root polynomial contracts, transcript adapters and sumcheck orchestration remain
consumer abstractions. Their operations delegate to shared arithmetic. The
sumcheck test hook also admits an independent bigint oracle; it is not a second
production backend. Storage, matrix layouts, NTT plans, Wengert plans, packed
readers, grinding boundaries and `U32MulBitzWidth` remain consumer-owned.

## Protocol boundaries

Shared canonical codecs and bounded public prime sampling are integrated.
Absorption is length-framed, internal squeezing advances, intervals/policies are
bound and prover/verifier replay the same search. Accepted prime preparation is
reused. Internal rejection reads do not add grinding boundaries. Trusted runtime
constructors use debug-only primality checks; untrusted configuration validation
and protocol replay remain checked. There is no compile-time Miller–Rabin.

Spartan transcript domains use v2; ECDSA encoding uses `BITZSE03` and hybrid
encoding uses version 6. Protocol parameters, field definitions and opening modes
are unchanged. Transcript fixtures are regenerated only from proofs that verify.
No new whole-proof security theorem is claimed.

## Correctness and remaining qualification

Generated validation ledgers and logs have been removed from the repository.
Fresh validation must record its commands, source revision, and remaining gaps
in local output directories.
The [type inventory](../../outputs/unified-arithmetic-type-removal.md) records
removals; the [timing ledger](../../outputs/unified-arithmetic-timing-repairs.md)
records repairs requiring separate measurement.

Both ARM A/A qualification attempts expired at the idle gate with zero
samples. The latest attempt observed at most 84% idle against the required 88%
and expired after 600 seconds. Its environment is unqualified. The finite
5×32 → 5×64 → 10×128 ladder and fresh confirmation remain required. Consumer workloads, preparation,
allocations, verification and peak memory need their own complete gates.

Native x86 execution on `will` remains pending source-transfer approval following
an earlier automatic approval rejection. No source has been transferred.
Portable x86, PCLMUL/SSE and VPCLMUL cross-compilation pass; these checks establish
compilation only. No speedup, regression-free result or measured timing exception
is claimed. These outstanding acceptance requirements prevent marking the full
approved plan complete.
