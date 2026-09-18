// Test-only independent BigUint oracle, frozen before the bounded migration.
//! Faithful port of Limber's wired MultiSwap integer Mod-R1CS circuit.
//!
//! Provenance: `lucasxia01/limber-impl`, `benches/multiswap_modp.rs` (MIT),
//! the circuit behind Table 1 of "Limber: Low Overhead SNARKs for Integers
//! from Any PCS" (ePrint 2026/1635).  The constants, row/column allocation
//! order, witness advice, and per-row modulus schedule are reproduced
//! exactly so an BitZ proof of this relation is a proof of the same 6209-row
//! statement Limber benchmarks (`k = 0`, the only configuration Limber's
//! authors mark quotable).
//!
//! The relation is integer Mod-R1CS
//!
//! ```text
//! A·z ∘ B·z = C·z + mods ∘ quos      over Z,  z = (w, 1),
//! ```
//!
//! with one per-row modulus: `mods[r] >= 2` is an ordinary modular row
//! `LC_A·LC_B ≡ LC_C (mod mods[r])` with the prover quotient `quos[r]` as
//! advice, and `mods[r] = 0` is an exact integer row (the `m·q` term
//! vanishes; used for bit constraints `b·b = b`).  `mods[r] = 1` is
//! degenerate and never generated.  All witness and quotient values are
//! nonnegative and bounded by `2^2048` (checked at construction).
//!
//! MultiSwap itself (Ozdemir–Wahby–Whitehat–Boneh, USENIX Security 2020)
//! verifies a batch of swaps against an RSA accumulator via two Wesolowski
//! proofs.  The wired circuit contains 4 real square-and-multiply chains
//! with 352-bit exponents mod RSA-2048 (bit decomposition, binaryness, and
//! reconstruction rows included), 2 group multiplications, 4 wired
//! Pocklington certificate chains over Mersenne moduli, 3 chained
//! Poseidon-cost permutations mod the BLS12-381 scalar prime, 639
//! decomposition bit rows with 5 exact reconstruction rows, and one final
//! mod-`l` reduction row.

use blake3::Hasher;
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::Zero;
use thiserror::Error;

/// Upper bound (bits) on every witness and quotient value.
///
/// Every value in the wired circuit is reduced modulo one of the circuit
/// moduli (all `< 2^2048`) or is a quotient of a product of two such values
/// by its modulus, so `2^2048` bounds both blocks.  The BitZ commitment
/// stores exactly this many little-endian bits per assignment entry.
pub const MULTISWAP_VALUE_BITS: usize = 2048;

/// Base-hash model: rows charged per `H` invocation (`k > 0` only).
const H_ROWS: usize = 8;
/// Wired Pocklington (`Hp`) certificate chains and exponent bits per chain.
const HP_EXPS: usize = 4;
const HP_EXP_BITS: usize = 50;
/// Poseidon x^5 S-box cost: three multiplication rows per S-box.
const POSEIDON_ROWS_PER_PERM: usize = 243;
/// Poseidon permutations charged inside one `Hp` invocation.
const HP_POSEIDON_PERMS: usize = 3;
/// Bit rows decomposing the Poseidon output (255 bits) and the four chain
/// outputs (61 + 89 + 107 + 127 bits).  Fixed by the value widths.
const HP_DECOMP_BITS: usize = 639;
/// One exact reconstruction row per decomposed value.
const HP_DECOMP_RECON: usize = 5;
/// Group exponentiations per MultiSwap proof.
const N_GROUP_EXPS: usize = 4;
/// Group multiplications per MultiSwap proof.
const N_GROUP_MULS: usize = 2;
/// Exponent bit length of the Fiat--Shamir prime challenge `l`.
const ELL_BITS: usize = 352;

const CIRCUIT_DIGEST_DOMAIN: &[u8] = b"bitz/multiswap/circuit-digest/v1";

/// Row/column dimension schedule of one wired MultiSwap instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultiswapDims {
    poseidon_rows: usize,
    decomp_bits: usize,
    decomp_recon: usize,
    k: usize,
    ell_bits: usize,
    n_group_exps: usize,
    n_group_muls: usize,
    hp_exps: usize,
    hp_exp_bits: usize,
    h_rows: usize,
}

impl MultiswapDims {
    /// Limber's benchmark dimensions.  `k = 0` is the fully wired, quotable
    /// configuration (6209 rows); `k > 0` adds Limber's own unfaithful
    /// generic per-swap rows and is provided for parity only.
    pub const fn multiswap(k: usize) -> Self {
        Self {
            k,
            poseidon_rows: HP_POSEIDON_PERMS * POSEIDON_ROWS_PER_PERM,
            decomp_bits: HP_DECOMP_BITS,
            decomp_recon: HP_DECOMP_RECON,
            ell_bits: ELL_BITS,
            n_group_exps: N_GROUP_EXPS,
            n_group_muls: N_GROUP_MULS,
            hp_exps: HP_EXPS,
            hp_exp_bits: HP_EXP_BITS,
            h_rows: H_ROWS,
        }
    }

    /// Reduced-size dimensions with every wired row species present.
    ///
    /// Exercises the identical builder on a `2^11`-gate shape so tests do
    /// not pay for four 352-bit RSA-2048 exponentiations.  Not a Limber
    /// configuration; never quote it.
    pub const fn mini() -> Self {
        Self {
            k: 0,
            poseidon_rows: POSEIDON_ROWS_PER_PERM / 3,
            decomp_bits: HP_DECOMP_BITS,
            decomp_recon: HP_DECOMP_RECON,
            ell_bits: 40,
            n_group_exps: N_GROUP_EXPS,
            n_group_muls: N_GROUP_MULS,
            hp_exps: HP_EXPS,
            hp_exp_bits: 10,
            h_rows: H_ROWS,
        }
    }

    const fn rows_per_hp_exp(&self) -> usize {
        3 * self.hp_exp_bits + 1
    }

    const fn rows_per_exp(&self) -> usize {
        3 * self.ell_bits + 1
    }

    const fn cols_per_exp(&self) -> usize {
        3 * self.ell_bits + 1
    }

    const fn generic_rows(&self) -> usize {
        2 * self.k + 2 * self.k * self.h_rows
    }

    const fn wired_ext_rows(&self) -> usize {
        self.n_group_muls
            + self.hp_exps * self.rows_per_hp_exp()
            + 1
            + self.poseidon_rows
            + self.decomp_bits
            + self.decomp_recon
            + 1
    }

    const fn wired_ext_cols(&self) -> usize {
        self.n_group_muls
            + self.hp_exps * self.rows_per_hp_exp()
            + 1
            + self.poseidon_rows
            + self.decomp_bits
            + 1
    }

    /// Number of live (wired) constraint rows.
    pub const fn num_real_rows(&self) -> usize {
        self.n_group_exps * self.rows_per_exp() + self.generic_rows() + self.wired_ext_rows()
    }

    /// Number of live witness columns.
    pub const fn num_real_cols(&self) -> usize {
        self.n_group_exps * self.cols_per_exp() + 3 * self.generic_rows() + self.wired_ext_cols()
    }
}

/// Failures while constructing or checking the integer circuit.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum MultiswapCircuitError {
    #[error("batch count must be a power of two in 1..=16; batches require k=0")]
    InvalidBatch,

    /// A witness or quotient value exceeds [`MULTISWAP_VALUE_BITS`] bits.
    #[error("multiswap value at {location} column/row {index} needs {actual_bits} bits")]
    ValueTooWide {
        location: &'static str,
        index: usize,
        actual_bits: u64,
    },

    /// A generated matrix coordinate is outside the declared shape.
    #[error("multiswap matrix entry ({row}, {column}) is out of shape")]
    EntryOutOfShape { row: usize, column: usize },

    /// The integer relation `Az ∘ Bz = Cz + mods ∘ quos` fails at a row.
    #[error("multiswap integer relation fails at row {row}")]
    Unsatisfied { row: usize },
}

/// The complete integer Mod-R1CS statement and witness of one instance.
///
/// Matrices are raw COO triples over the Limber column space: witness
/// columns `0..num_vars` followed by the constant-one column `num_vars`
/// (`num_io = 0`).  `mods` has one entry per padded row; padding rows have
/// no matrix entries and modulus `2`, exactly as Limber pads.
#[derive(Clone, Debug)]
pub struct MultiswapCircuit {
    dims: MultiswapDims,
    batch_count: usize,
    num_cons: usize,
    num_vars: usize,
    a: Vec<(usize, usize, BigUint)>,
    b: Vec<(usize, usize, BigUint)>,
    c: Vec<(usize, usize, BigUint)>,
    mods: Vec<BigUint>,
    w: Vec<BigUint>,
    quos: Vec<BigUint>,
}

/// RSA-2048 modulus `N` (the RSA-2048 factoring-challenge value Limber
/// hardcodes).
pub fn modulus_n() -> BigUint {
    let hex = "c7970ceedcc3b0754490201a7aa613cd73911081c790f5f1a8726f463550bb5b\
               7ff0db8e1ea1189ec72f93d1650011bd721aeeacc2acde32a04107f0648c2813\
               a31f5b0b7765ff8b44b4b6ffc93384b646eb09c7cf5e8592d40ea33c80039f35\
               b4f14a04b51f7bfd781be4d1673164ba8eb991c2c4d730bbbe35f592bdef524a\
               f7e8daefd26c66fc02c479af89d64d373f442709439de66ceb955f3ea37d5159\
               f6135809f85334b5cb1813addc80cd05609f10ac6a95ad65872c909525bdad32\
               bc729592642920f24c61dc5b3c3b7923e56b16a4d9d373d8721f24a3fc0f1b31\
               31f55615172866bccc30f95054c824e733a5eb6817f7bc16399d48c6361cc7e5";
    BigUint::parse_bytes(hex.as_bytes(), 16).expect("valid RSA-2048 hex")
}

/// The 352-bit Fiat--Shamir challenge modulus `l`.
pub fn modulus_ell() -> BigUint {
    BigUint::from_bytes_be(&[0xc3u8; 44])
}

/// BLS12-381 scalar prime standing in for the Poseidon field.
pub fn modulus_p_hash() -> BigUint {
    let hex = "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001";
    BigUint::parse_bytes(hex.as_bytes(), 16).expect("valid BLS12-381 scalar hex")
}

/// Mersenne moduli of the wired `Hp` certificate chains.
#[allow(clippy::arithmetic_side_effects)]
fn hp_moduli() -> [BigUint; 4] {
    [
        (BigUint::from(1u32) << 61) - 1u32,
        (BigUint::from(1u32) << 89) - 1u32,
        (BigUint::from(1u32) << 107) - 1u32,
        (BigUint::from(1u32) << 127) - 1u32,
    ]
}

/// Deterministic `(base, exponent)` pairs for the `Hp` chains.
#[allow(clippy::arithmetic_side_effects)]
fn hp_chain_inputs(bits: usize) -> [(BigUint, BigUint); 4] {
    let ms = hp_moduli();
    core::array::from_fn(|i| {
        let base = &ms[i] - BigUint::from(1000u32 + 37 * i as u32);
        let exponent = (BigUint::from(0x9e37_79b9_7f4a_7c15u64) >> (64 - bits)) ^ BigUint::from(i);
        (base, exponent)
    })
}

#[allow(clippy::arithmetic_side_effects)]
fn exp_bases() -> [BigUint; 4] {
    let n = modulus_n();
    core::array::from_fn(|i| &n - BigUint::from(37u64 * i as u64 + 3))
}

#[allow(clippy::arithmetic_side_effects)]
fn exp_exponents(ell_bits: usize) -> [BigUint; 4] {
    core::array::from_fn(|i| {
        let seed = (i as u64 + 1) * 0x0123_4567_89AB_CDEFu64;
        let mut bytes = vec![0u8; ell_bits.div_ceil(8)];
        for (k, byte) in bytes.iter_mut().enumerate() {
            *byte = ((seed.wrapping_mul(k as u64 + 1).wrapping_add(0xDEAD)) & 0xFF) as u8;
        }
        if !ell_bits.is_multiple_of(8) {
            bytes[0] &= (1u8 << (ell_bits % 8)) - 1;
        }
        let msb_byte = (ell_bits - 1) / 8;
        let msb_bit = (ell_bits - 1) % 8;
        let msb_idx = bytes.len() - 1 - msb_byte;
        bytes[msb_idx] |= 1u8 << msb_bit;
        BigUint::from_bytes_be(&bytes)
    })
}

/// One wired square-and-multiply chain: square rows, conditional-multiply
/// rows (base folded into the `B` coefficients), exact binaryness rows, and
/// one reconstruction row binding the bits to the exponent column.
#[allow(clippy::arithmetic_side_effects)]
#[allow(clippy::too_many_arguments)]
fn build_exp_circuit(
    base: &BigUint,
    exponent: &BigUint,
    n: &BigUint,
    ell_bits: usize,
    row_base: usize,
    col_base: usize,
    const_col: usize,
    a_entries: &mut Vec<(usize, usize, BigUint)>,
    b_entries: &mut Vec<(usize, usize, BigUint)>,
    c_entries: &mut Vec<(usize, usize, BigUint)>,
    mods: &mut Vec<BigUint>,
    w: &mut [BigUint],
    quos: &mut [BigUint],
) -> usize {
    let one = BigUint::from(1u32);
    let g_minus_1 = base - &one;

    let bit_col = |j: usize| col_base + j;
    let exp_col = col_base + ell_bits;
    let acc_col = |j: usize| col_base + ell_bits + 1 + j;
    let sq_col = |j: usize| col_base + 2 * ell_bits + 1 + j;

    let bits: Vec<u8> = (0..ell_bits)
        .map(|j| {
            let bit_pos = ell_bits - 1 - j;
            u8::from(exponent.bit(bit_pos as u64))
        })
        .collect();

    for j in 0..ell_bits {
        w[bit_col(j)] = BigUint::from(bits[j]);
    }
    w[exp_col] = exponent.clone();

    let mut row = row_base;
    for j in 0..ell_bits {
        let acc_val = if j == 0 {
            one.clone()
        } else {
            w[acc_col(j - 1)].clone()
        };

        // Square row.
        let sq_prod = &acc_val * &acc_val;
        let (sq_q, sq_val) = sq_prod.div_rem(n);
        w[sq_col(j)] = sq_val.clone();
        quos[row] = sq_q;

        let acc_j_col = if j == 0 { const_col } else { acc_col(j - 1) };
        a_entries.push((row, acc_j_col, one.clone()));
        b_entries.push((row, acc_j_col, one.clone()));
        c_entries.push((row, sq_col(j), one.clone()));
        mods.push(n.clone());
        row += 1;

        // Conditional-multiply row.
        let b_val = BigUint::from(bits[j]) * &g_minus_1 + &one;
        let cm_prod = &sq_val * &b_val;
        let (cm_q, acc_next) = cm_prod.div_rem(n);
        w[acc_col(j)] = acc_next;
        quos[row] = cm_q;

        a_entries.push((row, sq_col(j), one.clone()));
        b_entries.push((row, bit_col(j), g_minus_1.clone()));
        b_entries.push((row, const_col, one.clone()));
        c_entries.push((row, acc_col(j), one.clone()));
        mods.push(n.clone());
        row += 1;
    }

    // Binaryness as exact integer rows (modulus 0: `b·b = b` over Z).
    for j in 0..ell_bits {
        a_entries.push((row, bit_col(j), one.clone()));
        b_entries.push((row, bit_col(j), one.clone()));
        c_entries.push((row, bit_col(j), one.clone()));
        quos[row] = BigUint::from(0u32);
        mods.push(BigUint::from(0u32));
        row += 1;
    }

    // Reconstruction: sum of the bit columns weighted by powers of two
    // equals the exponent column (quotient fixed to zero, so exact).
    for j in 0..ell_bits {
        let power = BigUint::from(1u32) << (ell_bits - 1 - j);
        a_entries.push((row, bit_col(j), power));
    }
    b_entries.push((row, const_col, one.clone()));
    c_entries.push((row, exp_col, one.clone()));
    quos[row] = BigUint::from(0u32);
    mods.push(n.clone());
    row += 1;

    let expected = base.modpow(exponent, n);
    assert_eq!(
        w[acc_col(ell_bits - 1)],
        expected,
        "exponentiation circuit witness mismatch"
    );

    row - row_base
}

impl MultiswapCircuit {
    /// Builds the complete wired shape, witness, and quotient advice.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn build(dims: MultiswapDims) -> Result<Self, MultiswapCircuitError> {
        let n = modulus_n();
        let ell = modulus_ell();
        let p_hash = modulus_p_hash();
        let bases = exp_bases();
        let exponents = exp_exponents(dims.ell_bits);

        let num_cons = dims.num_real_rows().next_power_of_two();
        let num_vars = dims.num_real_cols().next_power_of_two();
        let const_col = num_vars;

        let mut a_entries = Vec::new();
        let mut b_entries = Vec::new();
        let mut c_entries = Vec::new();
        let mut mods = Vec::new();
        let mut w = vec![BigUint::from(0u32); num_vars];
        let mut quos = vec![BigUint::from(0u32); num_cons];
        let one = BigUint::from(1u32);

        for i in 0..dims.n_group_exps {
            build_exp_circuit(
                &bases[i],
                &exponents[i],
                &n,
                dims.ell_bits,
                i * dims.rows_per_exp(),
                i * dims.cols_per_exp(),
                const_col,
                &mut a_entries,
                &mut b_entries,
                &mut c_entries,
                &mut mods,
                &mut w,
                &mut quos,
            );
        }

        let mut row = dims.n_group_exps * dims.rows_per_exp();
        let mut col = dims.n_group_exps * dims.cols_per_exp();

        // Wired group multiplications of the exponentiation outputs.
        let exp_out = |i: usize| i * dims.cols_per_exp() + 2 * dims.ell_bits;
        for i in 0..dims.n_group_muls {
            let a_col = exp_out(2 * i);
            let b_col = exp_out(2 * i + 1);
            let (qi, ci) = (&w[a_col] * &w[b_col]).div_rem(&n);
            w[col] = ci;
            quos[row] = qi;
            a_entries.push((row, a_col, one.clone()));
            b_entries.push((row, b_col, one.clone()));
            c_entries.push((row, col, one.clone()));
            mods.push(n.clone());
            row += 1;
            col += 1;
        }

        // Wired Hp certificate chains over the Mersenne moduli.
        let hp_ms = hp_moduli();
        let hp_inputs = hp_chain_inputs(dims.hp_exp_bits);
        let mut hp_out_cols = [0usize; 4];
        for i in 0..dims.hp_exps {
            build_exp_circuit(
                &hp_inputs[i].0,
                &hp_inputs[i].1,
                &hp_ms[i],
                dims.hp_exp_bits,
                row,
                col,
                const_col,
                &mut a_entries,
                &mut b_entries,
                &mut c_entries,
                &mut mods,
                &mut w,
                &mut quos,
            );
            hp_out_cols[i] = col + 2 * dims.hp_exp_bits;
            row += dims.rows_per_hp_exp();
            col += dims.rows_per_hp_exp();
        }

        // Poseidon seed: reduce the first exponentiation output mod p_hash.
        let seed_col = col;
        {
            let (qi, ci) = w[exp_out(0)].div_rem(&p_hash);
            w[seed_col] = ci;
            quos[row] = qi;
            a_entries.push((row, exp_out(0), one.clone()));
            b_entries.push((row, const_col, one.clone()));
            c_entries.push((row, seed_col, one.clone()));
            mods.push(p_hash.clone());
            row += 1;
            col += 1;
        }

        // Chained Poseidon-cost rows mod p_hash (x², x⁴, x⁵ per S-box).
        let zero = BigUint::from(0u32);
        let mut x_col = seed_col;
        for _ in 0..(dims.poseidon_rows / 3) {
            let x = w[x_col].clone();
            let (q2, x2) = (&x * &x).div_rem(&p_hash);
            let (q4, x4) = (&x2 * &x2).div_rem(&p_hash);
            let (q5, x5) = (&x4 * &x).div_rem(&p_hash);
            w[col] = x2;
            a_entries.push((row, x_col, one.clone()));
            b_entries.push((row, x_col, one.clone()));
            c_entries.push((row, col, one.clone()));
            mods.push(p_hash.clone());
            quos[row] = q2;
            row += 1;
            w[col + 1] = x4;
            a_entries.push((row, col, one.clone()));
            b_entries.push((row, col, one.clone()));
            c_entries.push((row, col + 1, one.clone()));
            mods.push(p_hash.clone());
            quos[row] = q4;
            row += 1;
            w[col + 2] = x5;
            a_entries.push((row, col + 1, one.clone()));
            b_entries.push((row, x_col, one.clone()));
            c_entries.push((row, col + 2, one.clone()));
            mods.push(p_hash.clone());
            quos[row] = q5;
            row += 1;
            x_col = col + 2;
            col += 3;
        }
        let pos_out_col = x_col;

        // Wired decomposition of the Poseidon output and the four chain
        // outputs, each with an exact reconstruction row.
        let decomp_targets: Vec<(usize, usize)> = std::iter::once((pos_out_col, 255))
            .chain((0..4).map(|i| (hp_out_cols[i], [61usize, 89, 107, 127][i])))
            .collect();
        debug_assert_eq!(
            decomp_targets.iter().map(|&(_, bits)| bits).sum::<usize>(),
            dims.decomp_bits
        );
        for &(val_col, nbits) in &decomp_targets {
            let val = w[val_col].clone();
            let bit_base = col;
            for j in 0..nbits {
                let bit = u8::from(val.bit((nbits - 1 - j) as u64));
                w[col] = BigUint::from(bit);
                a_entries.push((row, col, one.clone()));
                b_entries.push((row, col, one.clone()));
                c_entries.push((row, col, one.clone()));
                mods.push(zero.clone());
                quos[row] = zero.clone();
                row += 1;
                col += 1;
            }
            for j in 0..nbits {
                let power = BigUint::from(1u32) << (nbits - 1 - j);
                a_entries.push((row, bit_base + j, power));
            }
            b_entries.push((row, const_col, one.clone()));
            c_entries.push((row, val_col, one.clone()));
            mods.push(zero.clone());
            quos[row] = zero.clone();
            row += 1;
        }

        // Final mod-l reduction row, wired to the Poseidon output.
        {
            let (qi, ci) = w[pos_out_col].div_rem(&ell);
            w[col] = ci;
            quos[row] = qi;
            a_entries.push((row, pos_out_col, one.clone()));
            b_entries.push((row, const_col, one.clone()));
            c_entries.push((row, col, one.clone()));
            mods.push(ell.clone());
            row += 1;
            col += 1;
        }

        // Per-swap H-delta models (`k > 0` only; unfaithful, kept for
        // Limber parity).
        let groups: Vec<(&BigUint, usize)> =
            vec![(&ell, 2 * dims.k), (&p_hash, 2 * dims.k * dims.h_rows)];
        let mut r = 0usize;
        for (modulus, count) in &groups {
            for _ in 0..*count {
                let a_val = *modulus - BigUint::from((r as u64 % 17) + 1);
                let b_val = *modulus - BigUint::from(((r as u64 * 7) % 19) + 2);
                let prod = &a_val * &b_val;
                let (qi, ci) = prod.div_rem(modulus);
                w[col] = a_val;
                w[col + 1] = b_val;
                w[col + 2] = ci;
                quos[row] = qi;
                a_entries.push((row, col, one.clone()));
                b_entries.push((row, col + 1, one.clone()));
                c_entries.push((row, col + 2, one.clone()));
                mods.push((*modulus).clone());
                row += 1;
                col += 3;
                r += 1;
            }
        }
        debug_assert_eq!(row, dims.num_real_rows());
        debug_assert_eq!(col, dims.num_real_cols());

        mods.resize(num_cons, BigUint::from(2u32));

        let circuit = Self {
            dims,
            batch_count: 1,
            num_cons,
            num_vars,
            a: a_entries,
            b: b_entries,
            c: c_entries,
            mods,
            w,
            quos,
        };
        circuit.validate_shape()?;
        Ok(circuit)
    }

    /// Combine independent copies in the canonical live-row/live-column order.
    /// Padding happens once: inter-copy padding would hide constraints from
    /// the quotient folding that operates on the live row prefix.
    pub fn build_batch(
        dims: MultiswapDims,
        batch_count: usize,
    ) -> Result<Self, MultiswapCircuitError> {
        if !batch_count.is_power_of_two() || batch_count > 16 || (batch_count > 1 && dims.k != 0) {
            return Err(MultiswapCircuitError::InvalidBatch);
        }
        let base = Self::build(dims)?;
        if batch_count == 1 {
            return Ok(base);
        }
        let rows = base.live_rows();
        let cols = base.live_columns();
        let num_cons = (rows * batch_count).next_power_of_two();
        let num_vars = (cols * batch_count).next_power_of_two();
        let expand = |entries: &[(usize, usize, BigUint)]| {
            (0..batch_count)
                .flat_map(|copy| {
                    entries.iter().map(move |(r, c, v)| {
                        (
                            r + copy * rows,
                            if *c == base.num_vars {
                                num_vars
                            } else {
                                c + copy * cols
                            },
                            v.clone(),
                        )
                    })
                })
                .collect()
        };
        let mut result = Self {
            dims,
            batch_count,
            num_cons,
            num_vars,
            a: expand(&base.a),
            b: expand(&base.b),
            c: expand(&base.c),
            mods: vec![BigUint::from(2u32); num_cons],
            w: vec![BigUint::zero(); num_vars],
            quos: vec![BigUint::zero(); num_cons],
        };
        for copy in 0..batch_count {
            result.mods[copy * rows..(copy + 1) * rows].clone_from_slice(&base.mods[..rows]);
            result.quos[copy * rows..(copy + 1) * rows].clone_from_slice(&base.quos[..rows]);
            result.w[copy * cols..(copy + 1) * cols].clone_from_slice(&base.w[..cols]);
        }
        result.validate_shape()?;
        Ok(result)
    }

    pub const fn batch_count(&self) -> usize {
        self.batch_count
    }

    pub const fn live_columns(&self) -> usize {
        self.dims.num_real_cols() * self.batch_count
    }

    /// Backend-independent statement contract. The legacy matrix digest stays
    /// unchanged for B=1; this envelope additionally binds roles and public IO.
    pub fn comparison_statement_digest(&self) -> [u8; 32] {
        let mut h = Hasher::new();
        h.update(b"bitz-limber/multiswap-statement/v2");
        h.update(&self.statement_digest());
        for v in [
            self.batch_count,
            0,
            MULTISWAP_VALUE_BITS,
            self.live_rows(),
            self.live_columns(),
        ] {
            h.update(&(v as u64).to_le_bytes());
        }
        h.update(b"public:matrices,moduli;private:witness,quotients;unsigned;constant:one;padding:zero-witness,zero-quotients,modulus-two");
        *h.finalize().as_bytes()
    }

    fn validate_shape(&self) -> Result<(), MultiswapCircuitError> {
        let columns = self.num_vars + 1;
        for entries in [&self.a, &self.b, &self.c] {
            for &(row, column, _) in entries {
                if row >= self.num_cons || column >= columns {
                    return Err(MultiswapCircuitError::EntryOutOfShape { row, column });
                }
            }
        }
        for (index, value) in self.w.iter().enumerate() {
            if value.bits() > MULTISWAP_VALUE_BITS as u64 {
                return Err(MultiswapCircuitError::ValueTooWide {
                    location: "witness",
                    index,
                    actual_bits: value.bits(),
                });
            }
        }
        for (index, value) in self.quos.iter().enumerate() {
            if value.bits() > MULTISWAP_VALUE_BITS as u64 {
                return Err(MultiswapCircuitError::ValueTooWide {
                    location: "quotient",
                    index,
                    actual_bits: value.bits(),
                });
            }
        }
        Ok(())
    }

    /// Dimension schedule this circuit was built from.
    pub const fn dims(&self) -> &MultiswapDims {
        &self.dims
    }

    /// Padded (power-of-two) constraint-row count.
    pub const fn num_cons(&self) -> usize {
        self.num_cons
    }

    /// Padded (power-of-two) witness-column count, excluding the constant.
    pub const fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Number of live wired rows.
    pub const fn live_rows(&self) -> usize {
        self.dims.num_real_rows() * self.batch_count
    }

    /// Limber column index of the constant-one entry.
    pub const fn const_col(&self) -> usize {
        self.num_vars
    }

    /// COO entries of `A` over the Limber column space.
    pub fn a_entries(&self) -> &[(usize, usize, BigUint)] {
        &self.a
    }

    /// COO entries of `B` over the Limber column space.
    pub fn b_entries(&self) -> &[(usize, usize, BigUint)] {
        &self.b
    }

    /// COO entries of `C` over the Limber column space.
    pub fn c_entries(&self) -> &[(usize, usize, BigUint)] {
        &self.c
    }

    /// Per-row moduli (`0` marks an exact integer row).
    pub fn mods(&self) -> &[BigUint] {
        &self.mods
    }

    /// Integer witness values, one per Limber witness column.
    pub fn witness(&self) -> &[BigUint] {
        &self.w
    }

    /// Integer quotient advice, one per padded row.
    pub fn quotients(&self) -> &[BigUint] {
        &self.quos
    }

    /// Checks `A·z ∘ B·z = C·z + mods ∘ quos` over the integers.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn is_sat_integer(&self) -> Result<(), MultiswapCircuitError> {
        let multiply = |entries: &[(usize, usize, BigUint)]| -> Vec<BigUint> {
            let mut out = vec![BigUint::zero(); self.num_cons];
            for (row, column, value) in entries {
                let z = if *column == self.num_vars {
                    &BigUint::from(1u32) * value
                } else {
                    value * &self.w[*column]
                };
                out[*row] += z;
            }
            out
        };
        let az = multiply(&self.a);
        let bz = multiply(&self.b);
        let cz = multiply(&self.c);
        for row in 0..self.num_cons {
            if &az[row] * &bz[row] != &cz[row] + &self.mods[row] * &self.quos[row] {
                return Err(MultiswapCircuitError::Unsatisfied { row });
            }
        }
        Ok(())
    }

    /// Replaces the quotient advice, for negative tests only.
    #[cfg(test)]
    pub(crate) fn with_quotients_for_tests(mut self, quos: Vec<BigUint>) -> Self {
        assert_eq!(quos.len(), self.num_cons);
        self.quos = quos;
        self
    }

    /// Canonical BLAKE3 digest of the complete integer statement.
    ///
    /// Binds the dimension schedule, padded shape, all three COO matrices,
    /// and the per-row moduli with length-prefixed frames.  The witness and
    /// quotients are deliberately excluded.
    pub fn statement_digest(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(CIRCUIT_DIGEST_DOMAIN);
        for value in [
            self.dims.k,
            self.dims.ell_bits,
            self.dims.n_group_exps,
            self.dims.n_group_muls,
            self.dims.hp_exps,
            self.dims.hp_exp_bits,
            self.dims.poseidon_rows,
            self.dims.decomp_bits,
            self.dims.decomp_recon,
            self.dims.h_rows,
            self.num_cons,
            self.num_vars,
            MULTISWAP_VALUE_BITS,
        ] {
            hasher.update(&(value as u64).to_le_bytes());
        }
        for entries in [&self.a, &self.b, &self.c] {
            hasher.update(&(entries.len() as u64).to_le_bytes());
            for (row, column, value) in entries {
                hasher.update(&(*row as u64).to_le_bytes());
                hasher.update(&(*column as u64).to_le_bytes());
                let bytes = value.to_bytes_le();
                hasher.update(&(bytes.len() as u64).to_le_bytes());
                hasher.update(&bytes);
            }
        }
        hasher.update(&(self.mods.len() as u64).to_le_bytes());
        for modulus in &self.mods {
            let bytes = modulus.to_bytes_le();
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(&bytes);
        }
        *hasher.finalize().as_bytes()
    }
}
