# Limber: Low Overhead SNARKs for Integers

> **Fork notice.** This is a fork of [lucasxia01/limber-impl](https://github.com/lucasxia01/limber-impl),
> the reference implementation of [Limber](https://eprint.iacr.org/2026/1635) by Chen, Xia, Nguyen and Bünz.
> Upstream is the source of truth; this fork is used for benchmarking and comparison work
> and tracks upstream `main`.

## Main benchmark command

The headline number is the MultiSwap benchmark (Table 1 of the paper).
All paper numbers are single-threaded with native CPU codegen:

```bash
RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" cargo bench --bench multiswap_modp
```

Prepend `BDPCS=1` to run the Brakedown (hash-based) instantiation instead of Hyrax.
The other benches and knobs are documented in [Reproducing the paper's numbers](#reproducing-the-papers-numbers) below.

Upstream `main` currently requires Rust 1.97 or newer (`rustup update stable`); older toolchains fail to build the
library with an E0658 error on `isolate_lowest_one`.

This fork also adds `examples/spartan_part_timing.rs`, a per-phase timing harness that splits the prover
into its Spartan-style $\mathbb{Z}_p$ sumcheck part and its mod-PCS part on the msshape configs:

```bash
RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" RUST_LOG=info \
  cargo run --release --example spartan_part_timing
```

---

In this repo, we build SNARKs for Integers over Mod R1CS, where constraints support arbitrary modular arithmetic.
It implements the protocol from the accompanying paper [*Limber: Low Overhead SNARKs for Integers from Any PCS*](https://eprint.iacr.org/2026/1635).

This repo is forked from [Microsoft Spartan2](https://github.com/Microsoft/Spartan2), and we accordingly build Limber-Spartan with various choices of underlying PCS, including Hyrax and Brakedown.

## Integer Mod-R1CS Arithmetization

The Integer Mod-R1CS relation is different from the standard R1CS relation, adding a modulus `m_i` and a quotient `u_i` per constraint:

```text
A·z ∘ B·z = C·z + m ∘ u        (over the bounded integers)
```

so one row is one modular multiplication.
This naturally and efficiently supports large and varying modular arithmetic gates.

## Fingerprinting

Following [Zaratan](https://eprint.iacr.org/2024/1548), proving this Integer Mod-R1CS relation over the integers can be reduced to proving it over a randomly sampled prime modulus $p$ (fingerprinting).
However, we now need to be able to construct a PCS that is able to commit to **integers** in one field $`\mathbb{F}_q`$ and open in another $`\mathbb{F}_p`$.
We call this an **integer mod-PCS**.

## Limber: An Integer Mod-PCS Compiler

Limber is a compiler from **almost any** field PCS to an integer mod-PCS with $o(1)$ multiplicative commitment overhead.
The only requirement on the underlying field is $q = \Omega(\lambda^2 \mu^2)$ for a $\mu$-variable polynomial and $\lambda$-bit security, so it works over small fields (64-bit). The compiled scheme inherits the assumptions and performance of whatever PCS it wraps.
It has two algorithms:

- **Commit.** An integer polynomial $f$ with hypercube evaluations bounded by $`T_f`$ is limb-split into a polynomial $`f_{\mathsf{limb}}`$ whose entries are bounded by a base bound $T$ (we use $T = 2^{64}$), cast into $`\mathbb{F}_q`$, and committed with the underlying PCS.
A batched range check (LogUp-GKR) proves every limb is below $T$.

- **Evaluate mod $p$.** To open $f$ at a point $`\vec r \in \mathbb{Z}_p^{\mu}`$ to $v$, a sumcheck reduces the claim to one about $`f_{\mathsf{limb}}`$, the prover sends the *integer* evaluation $y \in \mathbb{Z}$, and the verifier checks $y \equiv v \pmod p$.
What remains is proving $`y = f_{\mathsf{limb}}(\vec{r'})`$ over the integers, which is handled by our IntEval protocol.
  - **IntEval.** The verifier samples $s$ small primes $`p_i`$ and the prover opens $`f_{\mathsf{limb}}`$ at $`\vec r \bmod p_i`$ for each.
  IntEval partially evaluates $k$ variables at a time to avoid overflow: the partially evaluated polynomial is decomposed as $`a + p_i \cdot b`$, the prover commits $a$ and $b$ (each $2^{-k}$ the size of the previous layer), and the protocol recurses on $a$.

### Parameters

The benchmarks below use the following parameters (Table 4 of the paper):

| Parameter | Hyrax | Brakedown | Meaning |
| --- | ---: | ---: | --- |
| $q$ | $\approx 2^{256}$ | $\approx 2^{256}$ | Field characteristic of the underlying PCS (Tom-256 scalar field) |
| $T$ | $2^{64}$ | $2^{64}$ | Limb base bound; every limb of $`f_{\mathsf{limb}}`$ is range-checked to $[0, T)$ |
| $k$ | $9$ | $11$ | Variables partially evaluated per IntEval layer; each layer shrinks by $2^{k}$ |
| $s$ | $16$ | $30$ | Number of small CRT primes $`p_i`$ sampled by the IntEval verifier |
| Commitment overhead | $\approx 0.13\times$ | $\approx 0.07\times$ | Extra committed data relative to the witness |

The values shown are for the MultiSwap benchmark ($`T_f = 2^{2048}`$, $N = 2^{13}$ rows). The above parameters for the plain Spartan comparison ($`T_f = 2^{256}`$, $N = 2^{10} \text{-} 2^{14}$) are similar.

$s$ and the commitment overhead are derived from the other three parameters and the polynomial size (`IntEvalParams::derive`).
Larger $k$ lowers the commitment overhead at the cost of more CRT primes. See the paper for more details.

## Code layout

- `src/provider/pcs/integer_modpcs.rs` — the mod-PCS compiler: commit, evaluate mod $p$, and IntEval.
- `src/provider/pcs/commit_backend.rs` — the `CommitBackend` trait for underlying PCSs.
- `src/provider/pcs/hyrax_pc.rs`, `src/provider/pcs/brakedown/` — the two PCS instantiations: Hyrax over the Tom-256 curve and Brakedown over the Tom-256 scalar field.
- `src/logup_gkr.rs` — batched LogUp-GKR range check for the limbs and decomposed polynomials in IntEval.
- `src/dyn_prime.rs`, `src/sumcheck_modp.rs`, `src/polys_modp/` — runtime-modulus field for the Fiat–Shamir-sampled prime $p$, and the sumcheck/polynomial code running over it.
- `src/imod_spartan_modp.rs` — the SNARK driver tying the Spartan-style mod-PIOP to the mod-PCS; trait surface in `src/traits/mod_engine.rs`.

## Building and testing

Requires a stable Rust toolchain (edition 2024, so Rust 1.85 or newer).

```bash
cargo build --release
cargo test --release
```

CI additionally runs `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings`.

## Benchmarks

All benchmarks use [Criterion](https://github.com/bheisler/criterion.rs) and report setup / prove / verify times plus proof sizes.
Run with native CPU codegen:

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench --bench <name>
```

| Bench | What it measures |
| --- | --- |
| `imod_spartan_modp` | Limber-Spartan on various constraint counts |
| `spartan_synthetic` | Plain-Spartan baseline to compare to Limber |
| `multiswap_modp` | MultiSwap (RSA-accumulator verification circuit, [OWWB20](https://eprint.iacr.org/2019/1494)). `BDPCS=1` runs the Brakedown instantiation instead of Hyrax |
| `logup_gkr` | LogUp-GKR range proof in isolation |

## Results

All numbers are single-threaded (`RAYON_NUM_THREADS=1`) on a MacBook (Apple M4 Pro, 24 GB RAM, 14 cores); all baselines were re-run on the same machine.

### MultiSwap benchmark
The MultiSwap [OWWB20](https://eprint.iacr.org/2019/1494) benchmark is two RSA accumulator updates (4 Wesolowski exponentiations with 352-bit exponents modulo an RSA-2048 modulus) plus a Poseidon-based hash-to-prime evaluation.

Constraint counts for [Zinc+](https://eprint.iacr.org/2026/855) and Limber are integer gates; the others are ordinary R1CS constraints over a prime field.

| System | Constraints | Prove | Verify | Proof size |
| --- | ---: | ---: | ---: | ---: |
| Arkworks (emulated field arithmetic + Garuda) | 25.2 M | 231 s | 25 ms | 7.2 KB |
| MultiSwap [OWWB20](https://eprint.iacr.org/2019/1494) (xJsnark techniques + Groth16) | 6.2 M | 88.5 s | 2 ms | 0.2 KB |
| [Zinc+](https://eprint.iacr.org/2026/855) (concurrent work based on Brakedown) | 6,209 | 2.06 s | 514 ms | 1.2 MB |
| **Limber-Spartan (Hyrax)** | 6,209 | **1.32 s** | **39 ms** | **170 KB** |
| **Limber-Spartan (Brakedown)** | 6,209 | **1.18 s** | **45 ms** | 5.3 MB |

Limber demonstrates a 175×/196× (Hyrax/Brakedown) prover speedup over Arkworks and 67×/75× over the MultiSwap implementation.
Against Zinc+, the Hyrax prover is 1.6× faster, the verifier 13× faster, and the proof 7× smaller (170 KB vs. 1.2 MB).
The Brakedown prover is 1.7× faster and the verifier 11× faster than Zinc+, with a 4.4× larger proof (5.3 MB vs. 1.2 MB).

### Non-native overhead relative to native constraints
This experiment compares Limber against a plain-Spartan baseline with the same constraint and variable counts.
Each Limber gate is a random multiplication modulo the Tom-256 **base-field** modulus; the baseline proves native **scalar-field** gates.
Prover time includes witness generation and commitment.

| Constraints | Prove (Limber) | Prove (Spartan) | Ratio | Verify (Limber) | Verify (Spartan) | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^10 | 121 ms | 19.4 ms | 6× | 25.9 ms | 15.1 ms | 1.7× |
| 2^12 | 270 ms | 46.0 ms | 6× | 28.7 ms | 15.4 ms | 1.9× |
| 2^14 | 790 ms | 97.7 ms | 8× | 27.5 ms | 17.2 ms | 1.6× |

We demonstrate a low (6-8×) prover overhead for large 256-bit non-native gates compared to proving completely native constraints.

## Reproducing the paper's numbers

All numbers quoted in the paper are **single-threaded** (`RAYON_NUM_THREADS=1`) with native codegen (`RUSTFLAGS="-C target-cpu=native"`).

### MultiSwap table (Table 1 of the paper)

Both of the Limber rows prove the MultiSwap computation: 4 fully wired Wesolowski exponentiations with 352-bit exponents mod an RSA-2048 modulus and the hash-to-prime which we model with the right number of gates (~2000).

**Hyrax row**:

```bash
RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" cargo bench --bench multiswap_modp
```

This instantiation defaults to `k = 9`; override with `IMOD_K=<k>`.
Set `PSIZE=1` to print the proof size instead, and `KSWEEP=1` to sweep the IntEval reduction parameter `k`.

**Brakedown row**:

```bash
BDPCS=1 RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" cargo bench --bench multiswap_modp
```

This instantiation defaults to `k = 11` (faster than `k = 9` for the hash backend); override with `BDK=<k>`.

**Arkworks/Garuda baseline row**: measured with an external harness, [`bbuenz/rsa-exp-snark`](https://github.com/bbuenz/rsa-exp-snark) (GR1CS `RsaExpCircuit` via r1cs-std emulated field arithmetic; Garuda/Pari from [`alireza-shirzad/garuda-pari`](https://github.com/alireza-shirzad/garuda-pari), BLS12-381).
The circuit contains ~25.2M constraints; reproducing the full-size row needs ~14 GB of RAM (single-threaded: keygen ~904 s, prove ~231 s, verify ~25 ms, proof 7.2 KB).

**Zinc+ row**:
We compare to Zinc+'s implementation [`NethermindEth/zinc-plus`](https://github.com/NethermindEth/zinc-plus) at commit `7eadc16` and with `crypto-primitives` git dependency pinned to rev `2cf39db8` to fix the build. Under this setup, run:

```bash
RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" \
  cargo bench --bench e2e --features "parallel simd unchecked iprs-rate-1-8"
```

Zinc+ does not benchmark big-integer arithmetic, so the 2048-bit workload is a small custom UAIR — one `a·b ≡ c (mod N)` per row via `assert_zero(a·b − c − k·N)` — written in their framework. We note that this circuit is not properly wired and it does not include the hash-to-prime or bit checking gates for the actual MultiSwap computation.

### Native-overhead figure (Figure 3 of the paper)
We use Limber-Spartan with Hyrax in this comparison. To generate the data and plots, run:
```bash
pip install matplotlib
RAYON_NUM_THREADS=1 ./scripts/regen_msshape_plots.sh
```

This runs the pair of benchmarks (`cargo bench --bench imod_spartan_modp -- msshape` vs `cargo bench --bench spartan_synthetic -- msshape`) and renders the figures via `scripts/plot_msshape.py`.
We get 5.9–8.1× prover overhead over plain Spartan at $2^{10}\text{–}2^{14}$ constraints, verify is under 30 ms vs 15–17 ms, proof is 135–149 KB vs ~68 KB.

## References
[Limber: Low Overhead SNARKs for Integers from Any PCS](https://eprint.iacr.org/2026/1635) — the protocol this repository implements.

[Spartan: Efficient and general-purpose zkSNARKs without trusted setup](https://eprint.iacr.org/2019/550) \
Srinath Setty \
CRYPTO 2020

[Scaling Verifiable Computation Using Efficient Set Accumulators](https://eprint.iacr.org/2019/1494) \
Alex Ozdemir, Riad S. Wahby, Barry Whitehat, Dan Boneh \
USENIX Security 2020

[Fully Succinct Arguments over the Integers from First Principles](https://eprint.iacr.org/2024/1548) (Zaratan) \
Matteo Campanelli, Mathias Hall-Andersen \
PKC 2026

[Zinc+: SNARKs for Polynomial Rings](https://eprint.iacr.org/2026/855) \
Alexander Abdugafarov, Albert Garreta, Amit Kumar, Michał Osadnik, Psi Vesely, Ilia Vlasov, Kai Zhe Zheng \
Cryptology ePrint Archive 2026/855

## License

MIT, inherited from the upstream [Spartan2](https://github.com/Microsoft/Spartan2) project — see [LICENSE](LICENSE).
