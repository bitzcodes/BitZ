# Shared arithmetic in vendor/field

Status: consumer migration implemented, acceptance in progress, 2026-09-15; integration base `6271724d`. The crate has moved here, the typed core is implemented, and scoped production consumers use it. [IMPLEMENTATION.md](IMPLEMENTATION.md) records completed checks and remaining work. Full performance qualification and native x86 execution remain pending. The user's later exception retains MultiSwap's local 192-bit low-`u128`/carry-word accumulator; its multiplication and reduction use shared arithmetic.

This is the canonical plan, superseding the location and interface sketches in `docs/unified-arithmetic-plan.md`. [API.md](API.md) records the type families, important signatures, implementation matrix, and coherence rules.

## Outcome and ownership

Build one maintained arithmetic package at `vendor/field`, retaining the Cargo package name `field`. BitZ, `crates/circuit`, and local `flock-core` depend on it. Move the existing `crates/field` package into that location; do not retain a forwarding package or a second numeric implementation.

The package owns fixed integers, binary fields, static/runtime prime fields, modular rings, typed wide products and accumulators, native mixed arithmetic, fused folds/rounds, preparation, and canonical codecs. Protocol schedules, transcripts, NTT plans, circuit expressions, sparse matrix structure, and proof types stay in their consumers.

Remove `vendor/crypto-primitives` and the migrated consumers' production dependencies on `crypto-primitives`, `crypto-bigint`, `crypto-primes`, `num-bigint`, and `num-integer` after all live users have replacements. Independent dev-only oracles remain allowed. ZKPassport, Limber, Plonky3, Spartan2, and Binius64 keep their native arithmetic and manifests.

The main performance requirement is to consume existing native or borrowed storage, combine conversion with the arithmetic that needs it, and write the representation required by the next consumer. Preserve preparation when reusing projected values is demonstrably cheaper than repeatedly operating on wide integers.

## Existing evidence and what it does not establish

- The frozen original shared crate lacked x86 GF128 acceleration and contained incomplete prime operations. The new crate has a PCLMUL adapter and typed arithmetic, but those additions do not establish native x86 or integrated performance.
- Historical Ryzen arithmetic repairs recorded 241 accepted candidate choices and 61 retained production choices across 302 cases. Retention is not a speedup. Generated run records are no longer tracked.
- A historical ARM replay recorded 32 selected single-thread cases and three ten-thread OOD cases passing the existing gate. This subset is not complete ARM coverage of the new library; fresh measurements are required.
- The two-limb fast MAC is wrapping modulo 2^128. It cannot stand in for an arbitrary exact five-limb accumulator.
- Public-input sparse/zero shortcuts do not establish private-input constant-time performance. A kernel's evidence applies to its actual input contract, source, executable, host, and workload.
- Moving a winning experiment behind a new API requires fresh measurements. Neither these reports nor the previous compiler-only trait probes establish an integrated prover speedup.

Use the supplied PIOP profile for prioritization: inner sumcheck 34.69 ms, matrix combination 6.95 ms, outer sumcheck 6.32 ms, terminal matrix evaluation 3.45 ms. Prime selection was 0.18 ms out of 53.52 ms; implement it correctly without making it a separate optimization project.

## Package layout and dependency cutover

Target layout:

```text
vendor/field/
  Cargo.toml, Cargo.lock
  PLAN.md, API.md, README.md, VENDORED.md
  LICENSE-*                         applicable upstream license texts
  src/
    lib.rs, traits.rs, ct.rs, codec.rs
    integer/                       values, exact products, division, views
    binary/                        F2, GF8, GF128, B127, F2 polynomials
    prime/                         identities, representations, contexts, sampling
    modular/                       general modular rings and prepared reduction
    batch/                         MAC, mixed operands, folds, fused rounds
    arch/                          portable, aarch64, x86_64 kernels
  tests/                           arithmetic, codecs, layout, compile-fail cases
  benches/                         thin drivers over actual public/shared APIs
```

Do not put BitZ or Flock protocol types into these modules. Architecture vector types, unchecked MAC helpers, reduction parameters, and raw constructors remain private.

Use these dependency paths, distinguishing relocation of existing dependencies from later consumer integration:

| Consumer manifest | Dependency | Checkpoint |
|---|---|---|
| Root BitZ | `field = { path = "vendor/field" }` | Add during BitZ integration (4). |
| `crates/circuit` | `field = { path = "../../vendor/field" }` | Update existing path during relocation (1). |
| Arithmetic experiment | `field = { path = "../../vendor/field" }` | Update existing path during relocation (1). |
| Local `flock-core` | `arithmetic = { package = "field", path = "../../../field", features = ["serde"] }` | Add during Flock integration (5), after shared serde support exists. |

Flock's dependency alias avoids confusion with its own `field` module. Shared integer storage eventually makes the dependency necessary outside circuit's old `full` feature; retain the feature for actual optional circuit capabilities, not for access to basic integers.

Keep the separate existing Cargo roots and Flock workspace. Move the field lockfile and update affected live root/circuit/experiment locks deliberately. Do not restructure the repository into a new workspace or change comparator locks as part of this migration. Add the new package's build-artifact ignore rule when moving it.

Preserve optional Spongefish integration and add optional serde at the type-owning crate. Threading must retain the current consumer/kernel boundaries and public size thresholds; moving code must not introduce nested parallel work or activate it through an unrelated feature. Keep v1 on the current native/std targets rather than adding a no_std project.

## Implementation checkpoints

Each checkpoint is separately reviewable and records source/build hashes, validation, changed cases, and its promotion decision. No checkpoint is accepted solely because the overall average improved.

### 0. Freeze references and repair the measurement boundary

1. Freeze a baseline B with the current production arithmetic and the accepted experimental selections. Capture relevant dirty-state patches without modifying unrelated user work, source archives, locks, toolchain, CPU features/placement, threads, profile/LTO, seeds, cases, and executables.
2. Freeze the union of supported user/paper workloads, including u32/u64/u128, BabyBear, SHA-256, P-256/ECDSA, MultiSwap, direct/virtualized openings, and raw PCS exponents 20-30. Resolve any still-incomplete hybrid rate-1/8 prerequisite as separate work before its baseline. Unsupported cases are named, never silently omitted.
3. Preserve immutable historical reports and snapshots. Add new relocation/API measurements in new result directories.
4. Adapt the active experiment harness before deleting source it extracts. `run.py` now uses `vendor/field`; `build.rs` and benchmark modules still extract/include production arithmetic by path and textual markers. Switch active comparisons to the new API and hash-bound independent baseline support. Frozen legacy source can be built from its archive outside the production tree.
5. Keep old labels only in historical evidence/readers. The live final prover benchmark reports one `delayed-reduction` implementation.

Exit: baseline can be reproduced independently of later source cleanup; the required-case manifest and acceptance policy are fixed.

### 1. Relocate the existing package without arithmetic changes

Move `crates/field` to `vendor/field`, update live paths/harness inputs and affected locks, and preserve the current kernels and behavior for this checkpoint. Preserve applicable license headers and add a provenance ledger identifying imported sources, revisions, modifications, and measured kernel selections.

Keep root BitZ release/bench at fat LTO and one codegen unit. Preserve standalone Flock's measured thin-LTO bench profile and the experiment's fat-LTO profile. A dependency manifest does not control the consuming binary's profile. Give standalone field benchmarks explicit matching settings; do not count profile changes as arithmetic improvements.

Exit: existing consumers resolve the relocated package; existing checks and targeted relocation timings pass with unchanged codegen settings. New BitZ/Flock dependencies wait for their integration checkpoints.

### 2. Construct the typed arithmetic core

Implement the API contract in [API.md](API.md): integer storage/masks/codecs; binary values; static/runtime prime identities; one Montgomery prime-field scalar; native integer operands and integer-stored plain residues; modular contexts; exact products/accumulators; scale-aware reduction; checked/wrapping arithmetic; prepared public division and private inversion. Do not introduce plain-field or small-field scalar wrappers.

Prime sampling belongs here: bounded public candidate sampling, unbiased candidate/base selection, a whole-search Miller-Rabin error budget, and reuse of accepted-candidate preparation. No compile-time Miller-Rabin. Known static primes use validated specifications and tests.

Use concrete trait implementations and private shared helpers. Field identities tag accumulators as well as values. Runtime arithmetic uses explicit context operations such as field.mul(a,b). Keep runtime scalar tables free of per-element context/modulus storage while preserving their context association. Cached plain-residue tables retain their context in the owning witness/prepared container; ordinary integer operands need no field membership or range check. Add compile-fail/layout tests before consumers migrate.

Exit: portable arithmetic, encoding, identity/scale safety, full-width primes, signed/unsigned products, and exact-vs-wrapping behavior pass differential tests.

### 3. Import measured kernels through the real API

Move/adapt accepted ARM and x86 implementations behind the shared interfaces. Preserve portable fallbacks, scalar multiplication chains, prepared scalar multiplication, exact/small/wrapping integer specializations, SIMD tails, fused weighted rounds, and fold-plus-round/grid operations. Reuse implementation where contracts match; do not wrap external production numeric libraries as the final implementation.

Keep each measured scalar/batch/prepared selection separate. Never choose a wider ISA merely because it exists. ARM PMULL requires `aes`; x86 kernels must gate every instruction they use, including PCLMUL, SSE4.1 extraction where relevant, and VPCLMUL/AVX-512 plus any tail requirements. Native builds are the primary performance target, with compile-time ISA selection and portable/restricted-feature validation.

Measure actual shared API calls against frozen production and the accepted candidate/strongest alternative. Include setup, hot execution, allocations, and short/tail-heavy cases. Select kernels using public shape and available instructions, never fixture identity or private magnitude.

Exit: supported cases retain their accepted performance after the API boundary; missing/inconclusive results do not promote a replacement.

### 4. Migrate mixed arithmetic and fused projection in BitZ

| Path | Required change |
|---|---|
| u32 | Preserve borrowed native assignment/products and the existing mixed first round. Remove the production immediate-conversion branch during strategy consolidation. |
| u64 | Keep borrowed assignment; extend native outer-round/fold kernels to u64 operands and exact split-u64 u128 products. Remove the eager outer-table projection where these kernels win. |
| u128 | Borrow segmented x/y and split-u128 z storage. Add mixed u128/u256 MAC and first-fold kernels for assignment and outer products; write folded outputs directly. |
| MultiSwap | Use fixed public-width integer storage and prepared coefficients. Compare fused wide-input processing with projecting once and reusing residues over sparse matrix passes. Preserve the faster justified choice internally. |
| SHA-256/P-256 and other inner claims | Preserve packed bits, native small coefficients, signed differences, sparse/generated terms, and existing fused matrix/round traversal. |

The existing native outer path validates Az/Bz as 32-bit values; its u64 storage does not make it safe for wider operands. Widen products, signed differences, and accumulator bounds before reuse.

Preserve the inner native fold's plain canonical witness representation: field-weight x native becomes field-weight x integer-stored plain residues after folding. Store the residues in full-width integer buffers associated with their context, without another public field scalar type. Values generally cease to be small after a random challenge. Do not force all folded buffers into Montgomery form. Outer kernels may write Montgomery output directly where the next multiplication needs it.

No mandatory `Vec<Fp>` or `Vec<Uint<1>>` adapter pass is allowed merely to call the library. Indexed MAC and fold readers handle segmented/split storage and public padding. Reduction still occurs where mathematically required; conversion can be fused rather than removed algebraically.

Exit: each migrated path has identical arithmetic/proof validation and its own targeted consumer performance pass, including preparation and memory use.

### 5. Unify Flock and circuit consumers

Flock uses the shared `Gf8`/`Gf128` names directly, without compatibility aliases. BitZ's duplicate GF128 becomes the same shared type. Remove conversion vectors and now-duplicate trait implementations. Move standard arithmetic and serde implementations into the owning package and provide its optional serde feature before enabling Flock's new dependency; keep Flock-local protocol traits and its NTT scheduling in Flock.

Verify the AES GF8 embedding, GHASH polynomial/bit order, 16-byte GF128 size/alignment, canonical bytes, and actual serde/bincode framing. Adapt inverse-zero conventions explicitly at callers using the shared masked inverse result. Do not replace the independent upstream Flock test oracle with the local Flock consumer and create a dependency cycle/self-comparison.

Migrate circuit integers, public coefficient pools, checked preparation, fixed arenas, bounds certificates, and prepared witness execution. Coefficients are evaluated immediately; symbolic expressions represent witness dependence only. Preserve materialized and Wengert backends, exact security-bound/norm calculations, packed witnesses, and public-width schedules. Move P-256 quotient/remainder and inversion off variable-size external integers.

Exit: both integrated BitZ and standalone Flock checks/benches pass; consumers use the shared values without conversion bridges or performance-obscuring wrappers.

### 6. Consolidate and delete superseded production code

Delete `SpartanReductionStrategy`, all immediate/Barrett/crypto-bigint production selection, strategy arguments/fields, empty `ProveOptions`, and backend-named duplicate entrypoints/reducers. Use the shared batch/reduction capabilities through the canonical prover entrypoints. Do not introduce a one-variant enum or a production feature restoring the old choices.

Keep real distinctions: native versus field storage, standard versus univariate-skip protocol, direct versus virtualized openings. `PiopWitness` describes actual storage and supports borrowing. Update examples, flags, scripts, and live benchmark metadata; remove production strategy sweeps.

After all live consumers and the active harness are migrated, delete `vendor/crypto-primitives`, obsolete duplicate arithmetic, and migrated production bigint/primality dependencies. Clean unused features/re-exports and obsolete adapters. Retain independent dev-only oracles and frozen baseline archives without changing comparator libraries.

Exit: no old package path, deleted vendor dependency, duplicate numeric owner, or selectable arithmetic backend remains in the migrated production graph.

### 7. Verify constant-time behavior and the final integrated result

Apply required private-input timing repairs in identifiable changes and measure their costs against a migration snapshot U as well as baseline B. Review actual new binaries, not only earlier experimental assembly. Cover built-in hints and the full owned witness/prover path; custom callbacks retain an explicit caller timing contract.

Complete fixed-schedule private range/hint/relation validation before the shared generic invalid-witness boundary. Invalid attempts disclose no transcript prefix. Successful validation permits normal public transcript release before variable-time public prime/PoW searches; later failures retain only that already-public prefix. Private values, private randomness, internal hashes, magnitude trimming, zero skipping, and secret-indexed tables do not become public through a representation change.

Run one full final comparison against B, using targeted U comparisons for attribution. Verify proofs, setup, witness generation, proving, verification, witness-to-proof latency, proof size, and peak RSS across the frozen support manifest. ARM and Ryzen results remain separate.

Exit: correctness, supported-case completeness, timing requirements, dependency cleanup, and performance gates all pass, with measured exceptions documented as below.

## Regression gates and required tests

### Arithmetic and API

- Differential scalar/wide/batch/indexed/fused results against independent references; exact and wrapping paths tested separately.
- Zero/one/max values, signed minima, carry-heavy inputs, negative sums, full-width/padded primes, composites/even moduli, and nonunits.
- Exact quotient and remainder for wide dividends, checked narrowing, integer-to-field projection, and canonical codecs.
- GF8 embedding and binary polynomial correctness; SIMD boundary lengths, offsets, tails, sentinels, and alignment.
- Wrong static prime, wrong scale, and wrong product domain must fail to compile. Runtime elements are ordinary compact values: they may escape construction, and matching them to the originating runtime modulus is the explicit caller contract.
- Alias/blanket-impl coherence checks and layout assertions. The earlier seven compiler probes are preliminary evidence, not a substitute for retained crate tests.
- No private value-dependent invalidity exits or kernel selection; public shape/capacity checks occur before the hot batch.

### Performance policy

- Preserve the existing 1% median/P95/every-process and allocation gate per required case; never average away regressions.
- Use candidate-blind A/A qualification with the predeclared ladder 5×32 → 5×64 → 10×128 and a fresh confirmation before locking a measurement policy. `calibrate_unified.py` implements this for the 18 prime API cases, using 10 ms timed-batch calibration. Unqualified controls block candidate promotion; consumer cases require their own controls. Do not relax the 1% limits or discard cases.
- Require median and P95 upper 95% confidence bounds <= 1.01 and each confirmation-process median ratio <= 1.01. Record the strongest predeclared alternative. Require a demonstrated improvement before describing a replacement as faster.
- Report setup and one-shot costs separately from prepared hot execution; retain warmed allocation counts/bytes and end-to-end RSS. A moved reduction or allocation must not disappear from the measured boundary.
- Missing, inconclusive, or failed evidence blocks that replacement. Retain the known-good kernel or repair/retest without relaxing limits or changing fixtures after seeing results.
- Only a specifically demonstrated necessary constant-time repair can receive the previously agreed attributed exception. Lost acceleration, added materialization, profile drift, unsupported cases, verification failures, or OOM are not covered.

## Completion checklist

- One maintained numeric package exists at `vendor/field`; all intended consumers resolve it.
- API, licenses/provenance, examples, feature coverage, and standalone/integrated build instructions match the final code.
- Mixed field/native-integer arithmetic, integer-stored plain residues, and fused conversion survive integration; reusable wide projections remain where measured beneficial.
- Production has one delayed-reduction approach and no dependency on the removed vendor package.
- Independent references and historical evidence remain reproducible.
- The final report links exact source/binary/configuration evidence and identifies every retained, improved, or explicitly excepted case. It makes no blanket speedup or regression-free claim beyond that coverage.

## Reconciliation with the approved 2026-09-15 plan

The baseline is `6271724d`; preserve its canonical Wengert reverse pass,
weighted A/B/C application, geometric power runs, packed column reads,
structured/repeated SHA prefix factoring, and ECDSA native signed projection.
Keep the execution/storage types listed in the refreshed removal inventory.
Coefficient capacities are explicit type-level widths, with typed storage
segments rather than an automatically growing bigint or one maximum-width slot.

Current protocol parameters are fixed. No legacy API/proof compatibility is
required. Protocol migration must use length-framed absorption and advancing
squeezing, distinguish logical challenge/grinding boundaries from internal RNG
reads, bind both sampling interval endpoints and policy, and replay sampling.
Budget rejection exhaustion and whole-search composite acceptance separately;
these are not a new whole-proof security theorem. Untrusted values never enter
through the trusted debug-only primality constructor.

Private set-bit enumeration, secret-dependent shape work, sign branches, and
normalized witness lengths in the newly merged code require the narrowly
attributed timing repairs. A generic scalar replacement that discards the public
run/shape factoring does not satisfy the preservation requirement.
