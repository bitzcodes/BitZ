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
use circuit::integer_storage::UnsignedIntegerTable;
use field::{CanonicalCodec, CtEq, CtMask, CtSelect, IntegerOps, PreparedDivisor, Uint, WideMul};

#[cfg(test)]
#[path = "circuit_reference.rs"]
mod reference;

/// Public COO coordinates with coefficient storage segmented by declared width.
#[derive(Clone, Debug, Default)]
pub struct IntegerCoo {
    coordinates: Vec<(usize, usize)>,
    coefficients: UnsignedIntegerTable,
}
impl IntegerCoo {
    fn push<const L: usize>(&mut self, (row, column, value): (usize, usize, Uint<L>)) {
        self.coordinates.push((row, column));
        self.coefficients.push(value);
    }
    pub fn len(&self) -> usize {
        self.coordinates.len()
    }
    pub fn is_empty(&self) -> bool {
        self.coordinates.is_empty()
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (usize, usize, &[u64])> {
        self.coordinates
            .iter()
            .zip(self.coefficients.iter())
            .map(|(&(r, c), v)| (r, c, v))
    }
    pub(crate) fn copy_coefficient_to(&self, index: usize, output: &mut UnsignedIntegerTable) {
        self.coefficients.copy_row_to(index, output);
    }
    fn repeat(
        &self,
        count: usize,
        rows: usize,
        columns: usize,
        old_constant: usize,
        new_constant: usize,
    ) -> Self {
        let mut output = Self::default();
        for copy in 0..count {
            for (i, &(r, c)) in self.coordinates.iter().enumerate() {
                output.coordinates.push((
                    r + copy * rows,
                    if c == old_constant {
                        new_constant
                    } else {
                        c + copy * columns
                    },
                ));
                self.coefficients.copy_row_to(i, &mut output.coefficients);
            }
        }
        output
    }
}

fn public_hex<const L: usize>(hex: &str) -> Uint<L> {
    assert_eq!(hex.len(), 16 * L);
    Uint::from_words(core::array::from_fn(|i| {
        u64::from_str_radix(&hex[(L - i - 1) * 16..(L - i) * 16], 16).expect("valid public modulus")
    }))
}
fn read<const L: usize>(value: &Uint<32>) -> Uint<L> {
    let narrowed = value.checked_resize_ct();
    debug_assert!(
        narrowed.validity().declassify(),
        "generator violated its declared width"
    );
    *narrowed.value()
}
fn div_product<const L: usize>(
    a: &Uint<L>,
    b: &Uint<L>,
    divisor: &PreparedDivisor<L>,
) -> (Uint<32>, Uint<L>) {
    let product = IntegerOps.mul_wide(a, b);
    let (q, r) = divisor.div_rem_product_ct(&product);
    // Both operands are reduced: their product divided by the modulus is
    // below that modulus, hence below 2^(64 L). No secret overflow branch.
    let q = q.checked_resize_ct::<32>();
    debug_assert!(q.validity().declassify());
    (*q.value(), r)
}
fn div_value<const L: usize>(
    value: &Uint<32>,
    divisor: &PreparedDivisor<L>,
) -> (Uint<32>, Uint<L>) {
    divisor.div_rem_ct(value)
}

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

const CIRCUIT_DIGEST_DOMAIN: &[u8] = b"bitz/multiswap/circuit-digest/v2";

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
    a: IntegerCoo,
    b: IntegerCoo,
    c: IntegerCoo,
    mods: UnsignedIntegerTable,
    w: Vec<Uint<32>>,
    quos: Vec<Uint<32>>,
}

/// RSA-2048 modulus `N` (the RSA-2048 factoring-challenge value Limber
/// hardcodes).
pub fn modulus_n() -> Uint<32> {
    let hex = "c7970ceedcc3b0754490201a7aa613cd73911081c790f5f1a8726f463550bb5b\
               7ff0db8e1ea1189ec72f93d1650011bd721aeeacc2acde32a04107f0648c2813\
               a31f5b0b7765ff8b44b4b6ffc93384b646eb09c7cf5e8592d40ea33c80039f35\
               b4f14a04b51f7bfd781be4d1673164ba8eb991c2c4d730bbbe35f592bdef524a\
               f7e8daefd26c66fc02c479af89d64d373f442709439de66ceb955f3ea37d5159\
               f6135809f85334b5cb1813addc80cd05609f10ac6a95ad65872c909525bdad32\
               bc729592642920f24c61dc5b3c3b7923e56b16a4d9d373d8721f24a3fc0f1b31\
               31f55615172866bccc30f95054c824e733a5eb6817f7bc16399d48c6361cc7e5";
    public_hex(hex)
}

/// The 352-bit Fiat--Shamir challenge modulus `l`.
pub fn modulus_ell() -> Uint<6> {
    Uint::from_words([
        0xc3c3c3c3c3c3c3c3,
        0xc3c3c3c3c3c3c3c3,
        0xc3c3c3c3c3c3c3c3,
        0xc3c3c3c3c3c3c3c3,
        0xc3c3c3c3c3c3c3c3,
        0x00000000c3c3c3c3,
    ])
}

/// BLS12-381 scalar prime standing in for the Poseidon field.
pub fn modulus_p_hash() -> Uint<4> {
    let hex = "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001";
    public_hex(hex)
}

/// Hp moduli use the declared two-limb Mersenne domain (61..127 bits).
fn hp_moduli() -> [Uint<2>; 4] {
    [61, 89, 107, 127].map(|bits| Uint::from((1u128 << bits) - 1))
}
fn hp_chain_inputs(bits: usize) -> [(Uint<2>, Uint<1>); 4] {
    let ms = hp_moduli();
    core::array::from_fn(|i| {
        (
            ms[i].wrapping_sub(&Uint::from_u64(1000 + 37 * i as u64)),
            Uint::from_u64((0x9e37_79b9_7f4a_7c15u64 >> (64 - bits)) ^ i as u64),
        )
    })
}
fn exp_bases() -> [Uint<32>; 4] {
    let n = modulus_n();
    core::array::from_fn(|i| n.wrapping_sub(&Uint::from_u64(37 * i as u64 + 3)))
}
fn exp_exponents(ell_bits: usize) -> [Uint<6>; 4] {
    core::array::from_fn(|i| {
        let seed = (i as u64 + 1) * 0x0123_4567_89ab_cdefu64;
        let mut bytes = vec![0u8; ell_bits.div_ceil(8)];
        for (k, byte) in bytes.iter_mut().enumerate() {
            *byte = seed.wrapping_mul(k as u64 + 1).wrapping_add(0xdead) as u8;
        }
        if !ell_bits.is_multiple_of(8) {
            bytes[0] &= (1u8 << (ell_bits % 8)) - 1;
        }
        bytes[0] |= 1 << ((ell_bits - 1) % 8);
        let mut le = [0; 48];
        for (dst, src) in le.iter_mut().zip(bytes.iter().rev()) {
            *dst = *src;
        }
        IntegerOps.decode_public(&le).expect("six-limb exponent")
    })
}

/// One wired square-and-multiply chain: square rows, conditional-multiply
/// rows (base folded into the `B` coefficients), exact binaryness rows, and
/// one reconstruction row binding the bits to the exponent column.
#[allow(clippy::arithmetic_side_effects)]
#[allow(clippy::too_many_arguments)]
fn build_exp_circuit<const L: usize, const E: usize>(
    base: &Uint<L>,
    exponent: &Uint<E>,
    n: &Uint<L>,
    ell_bits: usize,
    row_base: usize,
    col_base: usize,
    const_col: usize,
    a_entries: &mut IntegerCoo,
    b_entries: &mut IntegerCoo,
    c_entries: &mut IntegerCoo,
    mods: &mut UnsignedIntegerTable,
    w: &mut [Uint<32>],
    quos: &mut [Uint<32>],
) -> usize {
    let one = Uint::<1>::ONE;
    let g_minus_1 = base.wrapping_sub(&Uint::ONE);
    let divisor = PreparedDivisor::new(*n).expect("public nonzero chain modulus");

    let bit_col = |j: usize| col_base + j;
    let exp_col = col_base + ell_bits;
    let acc_col = |j: usize| col_base + ell_bits + 1 + j;
    let sq_col = |j: usize| col_base + 2 * ell_bits + 1 + j;

    let bits: Vec<u8> = (0..ell_bits)
        .map(|j| {
            let bit_pos = ell_bits - 1 - j;
            exponent.bit(bit_pos).as_u64() as u8
        })
        .collect();

    for j in 0..ell_bits {
        w[bit_col(j)] = Uint::from_u64(bits[j] as u64);
    }
    w[exp_col] = exponent.zero_extend();

    let mut row = row_base;
    for j in 0..ell_bits {
        let acc_val = if j == 0 {
            Uint::<L>::ONE
        } else {
            read::<L>(&w[acc_col(j - 1)])
        };

        // Square row.
        let (sq_q, sq_val) = div_product(&acc_val, &acc_val, &divisor);
        w[sq_col(j)] = sq_val.zero_extend();
        quos[row] = sq_q;

        let acc_j_col = if j == 0 { const_col } else { acc_col(j - 1) };
        a_entries.push((row, acc_j_col, one.clone()));
        b_entries.push((row, acc_j_col, one.clone()));
        c_entries.push((row, sq_col(j), one.clone()));
        mods.push(n.clone());
        row += 1;

        // Conditional-multiply row.
        let b_val = Uint::ct_select(&Uint::ONE, base, CtMask::from_lsb(bits[j] as u64));
        let (cm_q, acc_next) = div_product(&sq_val, &b_val, &divisor);
        w[acc_col(j)] = acc_next.zero_extend();
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
        quos[row] = Uint::ZERO;
        mods.push(Uint::<1>::ZERO);
        row += 1;
    }

    // Reconstruction: sum of the bit columns weighted by powers of two
    // equals the exponent column (quotient fixed to zero, so exact).
    for j in 0..ell_bits {
        let power = Uint::<E>::ONE.truncating_shl(ell_bits - 1 - j);
        a_entries.push((row, bit_col(j), power));
    }
    b_entries.push((row, const_col, one.clone()));
    c_entries.push((row, exp_col, one.clone()));
    quos[row] = Uint::ZERO;
    mods.push(n.clone());
    row += 1;

    row - row_base
}

pub(super) fn public_coefficient_product(words: &[u64], value: &Uint<32>) -> Uint<65> {
    fn product<const L: usize>(words: &[u64], value: &Uint<32>) -> Uint<65> {
        let coefficient =
            Uint::<L>::from_words(words.try_into().expect("declared coefficient width"));
        *IntegerOps
            .mul_wide(&coefficient, value)
            .checked_resize_ct::<65>()
            .value()
    }
    match words.len() {
        1 => product::<1>(words, value),
        2 => product::<2>(words, value),
        4 => product::<4>(words, value),
        6 => product::<6>(words, value),
        32 => product::<32>(words, value),
        _ => unreachable!("circuit uses only its declared modulus and exponent widths"),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_generic_rows<const L: usize>(
    modulus: &Uint<L>,
    count: usize,
    start: usize,
    row: &mut usize,
    col: &mut usize,
    a: &mut IntegerCoo,
    b: &mut IntegerCoo,
    c: &mut IntegerCoo,
    mods: &mut UnsignedIntegerTable,
    w: &mut [Uint<32>],
    quos: &mut [Uint<32>],
) {
    let divisor = PreparedDivisor::new(*modulus).expect("public row modulus");
    for r in start..start + count {
        let av = modulus.wrapping_sub(&Uint::from_u64(r as u64 % 17 + 1));
        let bv = modulus.wrapping_sub(&Uint::from_u64((r as u64 * 7) % 19 + 2));
        let (q, cv) = div_product(&av, &bv, &divisor);
        w[*col] = av.zero_extend();
        w[*col + 1] = bv.zero_extend();
        w[*col + 2] = cv.zero_extend();
        quos[*row] = q;
        a.push((*row, *col, Uint::<1>::ONE));
        b.push((*row, *col + 1, Uint::<1>::ONE));
        c.push((*row, *col + 2, Uint::<1>::ONE));
        mods.push(*modulus);
        *row += 1;
        *col += 3;
    }
}

impl MultiswapCircuit {
    /// Builds the complete wired shape, witness, and quotient advice.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn build(dims: MultiswapDims) -> Result<Self, MultiswapCircuitError> {
        let n = modulus_n();
        let ell = modulus_ell();
        let p_hash = modulus_p_hash();
        let n_divisor = PreparedDivisor::new(n).expect("public RSA modulus");
        let hash_divisor = PreparedDivisor::new(p_hash).expect("public hash modulus");
        let ell_divisor = PreparedDivisor::new(ell).expect("public exponent modulus");
        let bases = exp_bases();
        let exponents = exp_exponents(dims.ell_bits);

        let num_cons = dims.num_real_rows().next_power_of_two();
        let num_vars = dims.num_real_cols().next_power_of_two();
        let const_col = num_vars;

        let mut a_entries = IntegerCoo::default();
        let mut b_entries = IntegerCoo::default();
        let mut c_entries = IntegerCoo::default();
        let mut mods = UnsignedIntegerTable::default();
        let mut w = vec![Uint::<32>::ZERO; num_vars];
        let mut quos = vec![Uint::<32>::ZERO; num_cons];
        let one = Uint::<1>::ONE;

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
            let (qi, ci) = div_product(&w[a_col], &w[b_col], &n_divisor);
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
            let (qi, ci) = div_value(&w[exp_out(0)], &hash_divisor);
            w[seed_col] = ci.zero_extend();
            quos[row] = qi;
            a_entries.push((row, exp_out(0), one.clone()));
            b_entries.push((row, const_col, one.clone()));
            c_entries.push((row, seed_col, one.clone()));
            mods.push(p_hash.clone());
            row += 1;
            col += 1;
        }

        // Chained Poseidon-cost rows mod p_hash (x², x⁴, x⁵ per S-box).
        let zero = Uint::<1>::ZERO;
        let mut x_col = seed_col;
        for _ in 0..(dims.poseidon_rows / 3) {
            let x = read::<4>(&w[x_col]);
            let (q2, x2) = div_product(&x, &x, &hash_divisor);
            let (q4, x4) = div_product(&x2, &x2, &hash_divisor);
            let (q5, x5) = div_product(&x4, &x, &hash_divisor);
            w[col] = x2.zero_extend();
            a_entries.push((row, x_col, one.clone()));
            b_entries.push((row, x_col, one.clone()));
            c_entries.push((row, col, one.clone()));
            mods.push(p_hash.clone());
            quos[row] = q2;
            row += 1;
            w[col + 1] = x4.zero_extend();
            a_entries.push((row, col, one.clone()));
            b_entries.push((row, col, one.clone()));
            c_entries.push((row, col + 1, one.clone()));
            mods.push(p_hash.clone());
            quos[row] = q4;
            row += 1;
            w[col + 2] = x5.zero_extend();
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
                let bit = val.bit(nbits - 1 - j).as_u64();
                w[col] = Uint::from_u64(bit);
                a_entries.push((row, col, one.clone()));
                b_entries.push((row, col, one.clone()));
                c_entries.push((row, col, one.clone()));
                mods.push(zero.clone());
                quos[row] = Uint::ZERO;
                row += 1;
                col += 1;
            }
            for j in 0..nbits {
                let power = Uint::<4>::ONE.truncating_shl(nbits - 1 - j);
                a_entries.push((row, bit_base + j, power));
            }
            b_entries.push((row, const_col, one.clone()));
            c_entries.push((row, val_col, one.clone()));
            mods.push(zero.clone());
            quos[row] = Uint::ZERO;
            row += 1;
        }

        // Final mod-l reduction row, wired to the Poseidon output.
        {
            let (qi, ci) = div_value(&w[pos_out_col], &ell_divisor);
            w[col] = ci.zero_extend();
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
        build_generic_rows(
            &ell,
            2 * dims.k,
            0,
            &mut row,
            &mut col,
            &mut a_entries,
            &mut b_entries,
            &mut c_entries,
            &mut mods,
            &mut w,
            &mut quos,
        );
        build_generic_rows(
            &p_hash,
            2 * dims.k * dims.h_rows,
            2 * dims.k,
            &mut row,
            &mut col,
            &mut a_entries,
            &mut b_entries,
            &mut c_entries,
            &mut mods,
            &mut w,
            &mut quos,
        );
        debug_assert_eq!(row, dims.num_real_rows());
        debug_assert_eq!(col, dims.num_real_cols());

        while mods.len() < num_cons {
            mods.push(Uint::<1>::from_u64(2));
        }

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
        let mut result = Self {
            dims,
            batch_count,
            num_cons,
            num_vars,
            a: base
                .a
                .repeat(batch_count, rows, cols, base.num_vars, num_vars),
            b: base
                .b
                .repeat(batch_count, rows, cols, base.num_vars, num_vars),
            c: base
                .c
                .repeat(batch_count, rows, cols, base.num_vars, num_vars),
            mods: UnsignedIntegerTable::default(),
            w: vec![Uint::ZERO; num_vars],
            quos: vec![Uint::ZERO; num_cons],
        };
        for copy in 0..batch_count {
            for r in 0..rows {
                base.mods.copy_row_to(r, &mut result.mods);
            }
            result.quos[copy * rows..(copy + 1) * rows].clone_from_slice(&base.quos[..rows]);
            result.w[copy * cols..(copy + 1) * cols].clone_from_slice(&base.w[..cols]);
        }
        while result.mods.len() < num_cons {
            result.mods.push(Uint::<1>::from_u64(2));
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

    /// BitZ protocol statement contract, including roles and public IO.
    pub fn comparison_statement_digest(&self) -> [u8; 32] {
        self.comparison_digest_with_statement(self.statement_digest())
    }

    /// Backend-independent contract for matched benchmark reports.
    /// Uses the minimal-byte matrix encoding shared with Limber, independently
    /// of the fixed-width encoding used by the BitZ proof transcript.
    pub fn canonical_comparison_statement_digest(&self) -> [u8; 32] {
        self.comparison_digest_with_statement(self.canonical_statement_digest())
    }

    fn comparison_digest_with_statement(&self, statement: [u8; 32]) -> [u8; 32] {
        let mut h = Hasher::new();
        h.update(b"bitz-limber/multiswap-statement/v2");
        h.update(&statement);
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
            for (row, column, _) in entries.iter() {
                if row >= self.num_cons || column >= columns {
                    return Err(MultiswapCircuitError::EntryOutOfShape { row, column });
                }
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
    pub fn a_entries(&self) -> &IntegerCoo {
        &self.a
    }

    /// COO entries of `B` over the Limber column space.
    pub fn b_entries(&self) -> &IntegerCoo {
        &self.b
    }

    /// COO entries of `C` over the Limber column space.
    pub fn c_entries(&self) -> &IntegerCoo {
        &self.c
    }

    /// Per-row moduli (`0` marks an exact integer row).
    pub fn mods(&self) -> &UnsignedIntegerTable {
        &self.mods
    }

    /// Integer witness values, one per Limber witness column.
    pub fn witness(&self) -> &[Uint<32>] {
        &self.w
    }

    /// Integer quotient advice, one per padded row.
    pub fn quotients(&self) -> &[Uint<32>] {
        &self.quos
    }

    /// Checks `A·z ∘ B·z = C·z + mods ∘ quos` over the integers.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn is_sat_integer(&self) -> Result<(), MultiswapCircuitError> {
        let multiply = |entries: &IntegerCoo| -> Vec<Uint<65>> {
            let mut out = vec![Uint::<65>::ZERO; self.num_cons];
            for (row, column, words) in entries.iter() {
                let w = if column == self.num_vars {
                    Uint::ONE
                } else {
                    self.w[column]
                };
                let term = public_coefficient_product(words, &w);
                // <2^64 products of two 32-limb values fit 65 limbs.
                out[row] = out[row].wrapping_add(&term);
            }
            out
        };
        let az = multiply(&self.a);
        let bz = multiply(&self.b);
        let cz = multiply(&self.c);
        let mut failure = self.num_cons as u64;
        for row in 0..self.num_cons {
            let lhs = IntegerOps.mul_wide(&az[row], &bz[row]);
            let rhs = public_coefficient_product(&self.mods[row], &self.quos[row]);
            let rhs = rhs
                .zero_extend::<130>()
                .wrapping_add(&cz[row].zero_extend());
            let equal = lhs.checked_resize_ct::<130>().value().ct_eq(&rhs);
            failure = u64::ct_select(
                &failure,
                &(row as u64),
                !equal & failure.ct_eq(&(self.num_cons as u64)),
            );
        }
        if failure == self.num_cons as u64 {
            Ok(())
        } else {
            Err(MultiswapCircuitError::Unsatisfied {
                row: failure as usize,
            })
        }
    }

    /// Replaces the quotient advice, for negative tests only.
    #[cfg(test)]
    pub(crate) fn with_quotients_for_tests(mut self, quos: Vec<Uint<32>>) -> Self {
        assert_eq!(quos.len(), self.num_cons);
        self.quos = quos;
        self
    }

    /// BitZ protocol BLAKE3 digest of the complete integer statement (v2).
    ///
    /// Binds the dimension schedule, padded shape, all three COO matrices,
    /// and the per-row moduli with length-prefixed frames.  The witness and
    /// quotients are deliberately excluded. Integers retain their storage width.
    pub fn statement_digest(&self) -> [u8; 32] {
        self.statement_digest_with_encoding(CIRCUIT_DIGEST_DOMAIN, false)
    }

    /// Canonical v1 digest for comparison with Limber's integer statement.
    /// Public coefficients and moduli use minimal unsigned little-endian bytes,
    /// with zero encoded as one zero byte, matching BigUint::to_bytes_le.
    /// This diagnostic encoding is not used by the BitZ proof transcript.
    pub fn canonical_statement_digest(&self) -> [u8; 32] {
        self.statement_digest_with_encoding(b"bitz/multiswap/circuit-digest/v1", true)
    }

    fn statement_digest_with_encoding(&self, domain: &[u8], minimal: bool) -> [u8; 32] {
        let encode = |words: &[u64]| {
            let mut bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            if minimal {
                // Only public statement values are formatted here.
                let len = bytes.iter().rposition(|&byte| byte != 0).map_or(1, |i| i + 1);
                bytes.truncate(len);
            }
            bytes
        };
        let mut hasher = Hasher::new();
        hasher.update(domain);
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
            for (row, column, value) in entries.iter() {
                hasher.update(&(row as u64).to_le_bytes());
                hasher.update(&(column as u64).to_le_bytes());
                let bytes = encode(value);
                hasher.update(&(bytes.len() as u64).to_le_bytes());
                hasher.update(&bytes);
            }
        }
        hasher.update(&(self.mods.len() as u64).to_le_bytes());
        for modulus in self.mods.iter() {
            let bytes = encode(modulus);
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(&bytes);
        }
        *hasher.finalize().as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_builder_matches_the_frozen_biguint_oracle() {
        let integer = |words: &[u64]| {
            num_bigint::BigUint::from_bytes_le(
                &words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        for full in [false, true] {
            let (dims, old_dims) = if full {
                (
                    MultiswapDims::multiswap(0),
                    reference::MultiswapDims::multiswap(0),
                )
            } else {
                (MultiswapDims::mini(), reference::MultiswapDims::mini())
            };
            let actual = MultiswapCircuit::build(dims).unwrap();
            let expected = reference::MultiswapCircuit::build(old_dims).unwrap();
            assert_eq!(actual.canonical_statement_digest(), expected.statement_digest());
            assert_eq!(
                actual.canonical_comparison_statement_digest(),
                expected.comparison_statement_digest()
            );
            if full {
                assert_eq!(
                    blake3::Hash::from(actual.statement_digest()).to_hex().as_str(),
                    "7b2a94147cfb9a0604c45f1ab13dacdefffb6474e26fffc8e376b147227523bf",
                    "preserve the existing BitZ protocol digest"
                );
                assert_eq!(
                    blake3::Hash::from(actual.canonical_statement_digest()).to_hex().as_str(),
                    "23e2a82b4c5aac1d8e7dcfd558c5a22e85b51efa20000a9fb1f0e1791a50a5da",
                    "match the independently recorded Limber digest"
                );
            }
            assert_eq!(actual.num_cons(), expected.num_cons());
            assert_eq!(actual.num_vars(), expected.num_vars());
            for (got, want) in actual.witness().iter().zip(expected.witness()) {
                assert_eq!(&integer(got.as_words()), want);
            }
            for (got, want) in actual.quotients().iter().zip(expected.quotients()) {
                assert_eq!(&integer(got.as_words()), want);
            }
            for (got, want) in actual.mods().iter().zip(expected.mods()) {
                assert_eq!(&integer(got), want);
            }
            for (got, want) in [
                (actual.a_entries(), expected.a_entries()),
                (actual.b_entries(), expected.b_entries()),
                (actual.c_entries(), expected.c_entries()),
            ] {
                assert_eq!(got.len(), want.len());
                for ((r, c, value), (er, ec, ev)) in got.iter().zip(want) {
                    assert_eq!((r, c), (*er, *ec));
                    assert_eq!(&integer(value), ev);
                }
                // Unit coefficients occupy one limb even in the RSA circuit.
                assert!(got.iter().any(|(_, _, v)| v.len() == 1));
            }
        }
    }

    #[test]
    fn canonical_batch_digests_match_limber_encoding_and_bind_public_statement() {
        let mut actual = MultiswapCircuit::build_batch(MultiswapDims::multiswap(0), 2).unwrap();
        let expected = reference::MultiswapCircuit::build_batch(
            reference::MultiswapDims::multiswap(0), 2,
        ).unwrap();
        assert_eq!(actual.canonical_statement_digest(), expected.statement_digest());
        let contract = actual.canonical_comparison_statement_digest();
        assert_eq!(contract, expected.comparison_statement_digest());
        let last = actual.live_rows() - 1;
        actual.quos[last] = actual.quos[last].wrapping_add(&Uint::ONE);
        assert_eq!(contract, actual.canonical_comparison_statement_digest());
        actual.mods.set(last, modulus_ell().wrapping_add(&Uint::ONE));
        assert_ne!(contract, actual.canonical_comparison_statement_digest());
    }

    #[test]
    fn batches_preserve_reference_and_reject_a_corrupt_last_copy() {
        let dims = MultiswapDims::multiswap(0);
        let reference = MultiswapCircuit::build(dims).unwrap();
        for batch in [1, 2, 4, 8, 16] {
            let mut circuit = MultiswapCircuit::build_batch(dims, batch).unwrap();
            circuit.is_sat_integer().unwrap();
            assert_eq!(circuit.live_rows(), 6209 * batch);
            assert_eq!(circuit.live_columns(), 6204 * batch);
            if batch == 1 {
                assert_eq!(circuit.statement_digest(), reference.statement_digest());
                assert_eq!(circuit.w, reference.w);
                assert_eq!(circuit.quos, reference.quos);
            }
            let before = circuit.comparison_statement_digest();
            // The last live row is modular, so changing its quotient changes
            // an actual equation rather than unconstrained padding advice.
            let last = circuit.live_rows() - 1;
            circuit.quos[last] = circuit.quos[last].wrapping_add(&Uint::ONE);
            assert_eq!(
                circuit.is_sat_integer(),
                Err(MultiswapCircuitError::Unsatisfied { row: last })
            );
            assert_eq!(
                before,
                circuit.comparison_statement_digest(),
                "witness is excluded from statement"
            );
            circuit
                .mods
                .set(last, modulus_ell().wrapping_add(&Uint::ONE));
            assert_ne!(before, circuit.comparison_statement_digest());
        }
        assert!(MultiswapCircuit::build_batch(dims, 0).is_err());
        assert!(MultiswapCircuit::build_batch(dims, 3).is_err());
        assert!(MultiswapCircuit::build_batch(dims, 32).is_err());
        assert!(MultiswapCircuit::build_batch(MultiswapDims::multiswap(1), 2).is_err());
    }

    #[test]
    fn quotable_dims_match_limber_row_and_column_counts() {
        let dims = MultiswapDims::multiswap(0);
        assert_eq!(dims.num_real_rows(), 6209);
        assert_eq!(dims.num_real_cols(), 6204);
        assert_eq!(dims.rows_per_exp(), 1057);
        assert_eq!(dims.rows_per_hp_exp(), 151);
    }

    #[test]
    fn mini_circuit_is_satisfied_over_the_integers() {
        let circuit = MultiswapCircuit::build(MultiswapDims::mini()).unwrap();
        assert_eq!(circuit.num_cons(), 2048);
        assert_eq!(circuit.num_vars(), 2048);
        circuit.is_sat_integer().unwrap();
    }

    #[test]
    fn tampered_witness_fails_the_integer_relation() {
        let mut circuit = MultiswapCircuit::build(MultiswapDims::mini()).unwrap();
        circuit.w[0] = circuit.w[0].wrapping_add(&Uint::ONE);
        assert!(matches!(
            circuit.is_sat_integer(),
            Err(MultiswapCircuitError::Unsatisfied { .. })
        ));
    }

    #[test]
    fn full_circuit_is_satisfied_and_has_limber_shape() {
        let circuit = MultiswapCircuit::build(MultiswapDims::multiswap(0)).unwrap();
        assert_eq!(circuit.num_cons(), 1 << 13);
        assert_eq!(circuit.num_vars(), 1 << 13);
        assert_eq!(circuit.live_rows(), 6209);
        circuit.is_sat_integer().unwrap();
    }

    #[test]
    fn statement_digest_is_stable_and_witness_independent() {
        let first = MultiswapCircuit::build(MultiswapDims::mini()).unwrap();
        let second = MultiswapCircuit::build(MultiswapDims::mini()).unwrap();
        assert_eq!(first.statement_digest(), second.statement_digest());
        let full = MultiswapCircuit::build(MultiswapDims::multiswap(0)).unwrap();
        assert_ne!(first.statement_digest(), full.statement_digest());
    }
}
