# Arithmetic provenance

The package was relocated from this repository's `crates/field` without changing
its arithmetic, from revision `c98675fe4028` plus the dirty state preserved in
a locally archived baseline snapshot.
The baseline manifest records individual source hashes and historical evidence.

`src/gf128/x86_64.rs` adapts the Karatsuba/Barrett multiplication and schoolbook
unreduced multiplication from
`vendor/flock-mod/crates/flock-core/src/field/gf2_128/x86_64.rs` in that snapshot.
The Binius/Irreducible and Succinct/Bunz/Wang attribution is retained. The source
is offered under Apache-2.0 OR MIT. The new adapter retains vector accumulators
and gates both PCLMULQDQ and SSE4.1, including the lane extraction requirement.

Existing ARM GF128 kernels retain their original provenance comments. Their
operation schedule has not been replaced by this relocation or typed API work.

The fixed-width integer and context API modules are new implementation in this
repository. Independent external libraries are used as test oracles; their code
is not copied into those modules.

`src/binary/gf8.rs` imports local Flock's `field/gf2_8.rs`, retaining its copyright
and Apache-2.0 OR MIT notice, with a masked portable multiplication path and
optional serde support. `src/binary/embedding.rs` contains its eight AES embedding
basis entries from `field/phi8.rs`; the secret-indexed 256-entry lookup is not used
by the new embedding API. The independent upstream Flock development dependency
remains a separate implementation for differential tests.

`src/binary/mod.rs` adapts the general B127 reduction from this repository's
`src/poly/univariate/binary_b127.rs`. `src/prime/barrett128.rs` and
`src/integer/kernels.rs` retain the fixed exact MAC and delayed-reduction schedules
from `src/utils/delayed_reduction.rs`. `src/prime/montgomery128.rs` adapts scalar
multiply/REDC from `src/piop/spartan/raw_monty.rs`, with masked final selection.
These sources are preserved in the baseline archive identified above. Arithmetic
parameter setup and public API adaptation are owned by this package. Importing a
schedule is not a claim that its performance gate has passed at the new call site.

`src/binary/b127_aarch64.rs` retains the canonical-product trinomial fold and
register-resident squaring chain from `src/poly/univariate/binary_b127.rs`.
The general fallback remains separate because arbitrary 256-bit polynomials
do not satisfy the canonical-product degree bound. AES and SHA3 instructions
have independent compile-time gates. `src/binary/prepared.rs` adapts the
interleaved fixed-scalar preparation and five-PMULL multiplication from
`src/poly/univariate/binary_gf128.rs` without its diagnostic environment switch.
`src/prime/dot.rs` extends the native inner-round and canonical witness-fold
formulas in `src/piop/spartan/raw_monty.rs` to declared native and limb widths.

`src/gf128/kernels/{aarch64,x86_64}.rs` imports complete local Flock kernel
modules at `6271724d`, including the four-lane VPCLMUL accumulator. They use the
shared scalar/product types. AVX512BW is included for per-lane byte shifts.
Flock continues to select its existing schedules through these shared entry
points; native performance qualification is outstanding. The ARM module also
owns the genus95 round-one NEON accumulator and half-width NTT twiddle multiply,
moved from local Flock's round1.rs and additive_ntt_f128.rs at the same base.

`src/prime/montgomery128.rs::mul_fios` retains the circuit Wengert evaluator's
FIOS schedule, with masked final correction. `src/prime/projection.rs` retains
ECDSA's canonical Horner projection and prepared 2^(64k) corrections from
`src/piop/spartan/raw_monty.rs`; sign correction now uses masks.

`src/modular/reduction.rs` is a new fixed-width radix Barrett division
implementation, including power-of-two divisors and public active-limb padding.
`PreparedOddInverse` uses a fixed binary extended-GCD schedule. Neither imports
an external production bigint implementation. The shared legacy Fq and comb
implementations have been consolidated into typed static primes and the generic
FixedBasePow provider; their historical source remains in the baseline archive.

`src/gf128/sumcheck.rs`, `src/gf128/kernels/sumcheck_aarch64.rs` and
`src/binary/b127_sumcheck.rs` retain BitZ's fused equality-weighted round and fold
kernels. `src/binary/projection.rs` retains the four-lane NEON packed projection
from BitZ's `binary_gf128.rs` at `6271724d`, using fixed public bounds and masked
private bits. Historical comparison copies are archived outside version control.
