#[cfg(test)]
use crate::piop::spartan::SpartanField as _;
#[cfg(test)]
use crate::piop::spartan::protocol::{FieldConfig as Config, SpartanBitzField as F};
use circuit::{
    integer_storage::IntegerTableView,
    matrix_products::IntegerProducts,
    p256, sha256,
    witgen::{PackedWitness, ProductWitgen, Witgen},
};
#[cfg(test)]
use num_bigint::BigInt;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::array;

use super::{
    Result, error,
    relation::{OuterMode, P_INPUT_ALIAS, PreparedSha256Ecdsa, SHA_F, SHA_H, Sha256EcdsaStatement},
};
use crate::{pcs::IntegerMatrixLayout, sumcheck::outer::OuterRows};

/// Packed source and virtual assignment, with exact P-256 row operands.
pub struct Sha256EcdsaWitness {
    pub(crate) f_rows: std::sync::Arc<Vec<Vec<u64>>>,
    pub(crate) h_rows: Vec<Vec<u64>>,
    pub(crate) products: IntegerProducts,
    pub(crate) statement: Sha256EcdsaStatement,
}

impl Sha256EcdsaWitness {
    /// Borrow exact P-256 rows; SHA and padding rows are structural zeros.
    pub(super) fn outer_integer_rows<'a>(
        &'a self,
        prepared: &'a PreparedSha256Ecdsa,
    ) -> impl OuterRows<AB = field::Z<5>, C = field::Z<9>> + 'a {
        let products = [
            &self.products.a_mw,
            &self.products.b_mw,
            &self.products.c_mw,
        ];
        // The P-256 circuit records signed nine-word rows. Check declared
        // storage widths, never private values, before entering the hot loop.
        assert!(
            products
                .iter()
                .all(|p| p.max_limbs() <= 9 && p.len() == prepared.local.rows())
        );
        let (offset, selection) = match prepared.mode {
            OuterMode::Split => (0, Some(prepared.local.nonlinear.as_slice())),
            OuterMode::AllRows => (256 * prepared.compressions(), None),
        };
        EcdsaOuterRows {
            products: products.map(|p| p.view()),
            selection,
            offset,
            count: selection.map_or(prepared.local.rows(), <[usize]>::len),
            rows: 1 << prepared.outer_sumcheck_num_vars(),
        }
    }

    /// MLE tables of `(A h) mod q`, `(B h) mod q`, and `(C h) mod q`: the
    /// independent BigInt projection used only by differential tests.
    #[cfg(test)]
    pub(super) fn build_outer_product_mles(
        &self,
        prepared: &PreparedSha256Ecdsa,
        q: u128,
        cfg: &Config,
    ) -> crate::piop::spartan::sumcheck::R1csProductMles<F> {
        use crate::poly::mle::DenseMultilinearExtension;
        let vars = prepared.outer_sumcheck_num_vars();
        let zero = F::zero_with_cfg(cfg);
        let mut tables = std::array::from_fn::<_, 3, _>(|_| vec![zero.clone(); 1 << vars]);
        let products = [
            &self.products.a_mw,
            &self.products.b_mw,
            &self.products.c_mw,
        ];
        for (table, products) in tables.iter_mut().zip(products) {
            let mut set = |dst: usize, src: usize| {
                let bytes: Vec<_> = products[src].iter().flat_map(|w| w.to_le_bytes()).collect();
                table[dst] =
                    super::reduce_integer_mod_q(&BigInt::from_signed_bytes_le(&bytes), q, cfg);
            };
            match prepared.mode {
                OuterMode::Split => {
                    for (i, &r) in prepared.local.nonlinear.iter().enumerate() {
                        set(i, r);
                    }
                }
                OuterMode::AllRows => {
                    for r in 0..prepared.local.rows() {
                        set(256 * prepared.compressions() + r, r);
                    }
                }
            }
        }
        // Honest SHA row products are identically zero. These are prover
        // claims, not a verifier assumption: the shared inner check binds the
        // zero C claims to the committed SHA assignment. Reuse them instead of
        // charging the comparison mode for a second SHA matrix multiplication.
        let [a, b, c] = tables.map(|evaluations| DenseMultilinearExtension {
            evaluations,
            num_vars: vars,
        });
        crate::piop::spartan::sumcheck::R1csProductMles {
            az: a,
            bz: b,
            cz: c,
        }
    }

    pub fn source_rows(&self) -> &[Vec<u64>] {
        &self.f_rows
    }
    pub fn assignment_rows(&self) -> &[Vec<u64>] {
        &self.h_rows
    }
    pub(crate) fn h_bit(&self, index: usize, p: &IntegerMatrixLayout) -> u64 {
        let row = index & (p.rows() - 1);
        (self.h_rows[index >> p.row_vars][row / 64] >> (row % 64)) & 1
    }
}

struct EcdsaOuterRows<'a> {
    products: [IntegerTableView<'a>; 3],
    selection: Option<&'a [usize]>,
    offset: usize,
    count: usize,
    rows: usize,
}
impl EcdsaOuterRows<'_> {
    #[inline]
    fn read<const N: usize>(&self, table: usize, row: usize) -> field::Z<N> {
        if row < self.offset || row - self.offset >= self.count {
            return field::Z::ZERO;
        }
        let row = row - self.offset;
        let source = self.selection.map_or(row, |indices| indices[row]);
        let words = &self.products[table][source];
        let sign = 0u64.wrapping_sub(words[words.len() - 1] >> 63);
        // build_local checks the public A/B row norms fit Z<5>; C retains
        // its declared nine limbs. Truncation here never depends on values.
        let mut extended = [sign; N];
        let count = words.len().min(N);
        extended[..count].copy_from_slice(&words[..count]);
        field::Z::from_twos_complement_words(extended)
    }
}
impl OuterRows for EcdsaOuterRows<'_> {
    type AB = field::Z<5>;
    type C = field::Z<9>;
    fn dimensions(&self) -> (usize, usize, usize) {
        (self.rows, self.rows, self.rows)
    }
    #[inline]
    fn a(&self, row: usize) -> Self::AB {
        self.read(0, row)
    }
    #[inline]
    fn b(&self, row: usize) -> Self::AB {
        self.read(1, row)
    }
    #[inline]
    fn c(&self, row: usize) -> Self::C {
        self.read(2, row)
    }
}

pub(crate) fn inverse(value: &[u8; 32]) -> Result<[u8; 32]> {
    use field::{CanonicalCodec, IntegerOps, Uint};
    let mut bytes = *value;
    bytes.reverse();
    let scalar: Uint<4> = IntegerOps
        .decode_public(&bytes)
        .expect("fixed-width scalar encoding");
    let inverse = p256::scalar_inverse_ct(&scalar);
    if !inverse.validity().declassify() {
        return Err(error("signature scalar is not in 1..n"));
    }
    IntegerOps.encode_into(inverse.value(), &mut bytes);
    bytes.reverse();
    Ok(bytes)
}

fn pack_bits(
    p: &IntegerMatrixLayout,
    live: usize,
    bit: impl Fn(usize) -> bool + Sync,
) -> Vec<Vec<u64>> {
    let column = |c: usize| {
        (0..p.rows().div_ceil(64))
            .map(|word| {
                let mut value = 0u64;
                for b in 0..64 {
                    let row = word * 64 + b;
                    let index = (c << p.row_vars) + row;
                    if row < p.rows() && index < live {
                        value |= u64::from(bit(index)) << b;
                    }
                }
                value
            })
            .collect()
    };
    #[cfg(feature = "parallel")]
    {
        (0..p.cols()).into_par_iter().map(column).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..p.cols()).map(column).collect()
    }
}

/// Copies `len` bits of little-endian packed `src` starting at bit `src_off`
/// into `dst` starting at bit `dst_off`; the destination bits must be zero.
fn copy_bits(dst: &mut [u64], dst_off: usize, src: &[u64], src_off: usize, len: usize) {
    let mut done = 0;
    while done < len {
        let (s, d) = (src_off + done, dst_off + done);
        let take = (64 - d % 64).min(len - done);
        let shift = s % 64;
        let mut value = src[s / 64] >> shift;
        // Assemble a whole destination word even when the source is unaligned.
        // Read the next source word only when the requested bits cross into it.
        if take > 64 - shift {
            value |= src[s / 64 + 1] << (64 - shift);
        }
        let mask = u64::MAX >> (64 - take);
        dst[d / 64] |= (value & mask) << (d % 64);
        done += take;
    }
}

/// In-place transpose of a 64×64 bit matrix: afterwards bit `r` of word `c`
/// is what bit `c` of word `r` was.
fn transpose64(a: &mut [u64; 64]) {
    let (mut j, mut m) = (32usize, 0x0000_0000_FFFF_FFFFu64);
    while j != 0 {
        let mut k = 0;
        while k < 64 {
            let t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k + j] ^= t;
            a[k] ^= t << j;
            k = (k + j + 1) & !j;
        }
        j >>= 1;
        m ^= m << j;
    }
}

/// The packed words of `witness` with the bits past its length cleared.
fn masked_word(witness: &PackedWitness, index: usize) -> u64 {
    let word = witness.words()[index];
    let valid = witness.bit_len() - 64 * index;
    if valid >= 64 {
        word
    } else {
        word & ((1u64 << valid) - 1)
    }
}

/// Copy across column boundaries directly into the final packed storage.
fn copy_column_bits(
    rows: &mut [Vec<u64>],
    row_vars: usize,
    mut dst: usize,
    src: &[u64],
    mut src_off: usize,
    mut len: usize,
) {
    let row_bits = 1usize << row_vars;
    while len != 0 {
        let offset = dst & (row_bits - 1);
        let take = len.min(row_bits - offset);
        copy_bits(&mut rows[dst >> row_vars], offset, src, src_off, take);
        dst += take;
        src_off += take;
        len -= take;
    }
}

/// The source rows: `f[0] = 1`, then each compression's block and hint bits,
/// then the P-256 source bits after its aliased inputs — word blits, no
/// per-bit closure.
fn pack_source(
    prepared: &PreparedSha256Ecdsa,
    shards: &[(PackedWitness, PackedWitness)],
    p_f: &PackedWitness,
) -> Vec<Vec<u64>> {
    let p = &prepared.f_layout;
    let mut rows = vec![vec![0u64; p.rows() / 64]; p.cols()];
    rows[0][0] = 1;
    for (instance, shard) in shards.iter().enumerate() {
        let dst = 1 + instance * SHA_F;
        copy_column_bits(&mut rows, p.row_vars, dst, shard.0.words(), 0, 512);
        copy_column_bits(
            &mut rows,
            p.row_vars,
            dst + 512,
            shard.0.words(),
            768,
            SHA_F - 512,
        );
    }
    copy_column_bits(
        &mut rows,
        p.row_vars,
        prepared.map.f_offset,
        p_f.words(),
        P_INPUT_ALIAS - 1,
        p_f.bit_len() - (P_INPUT_ALIAS - 1),
    );
    rows
}

/// The assignment rows `h[instance + N·local]`: every 64 consecutive cells are
/// one local wire across 64 compressions, so each 64×64 tile of (compression,
/// local) witness words is one bit-matrix transpose; the P-256 tail is a word
/// copy. Batches below 64 compressions keep the per-bit path.
fn pack_assignment(
    prepared: &PreparedSha256Ecdsa,
    shards: &[(PackedWitness, PackedWitness)],
    p_h: &PackedWitness,
) -> Vec<Vec<u64>> {
    let p = &prepared.h_layout;
    let n = prepared.compressions();
    if n < 64 {
        return pack_bits(p, prepared.live_assignment_bits(), |index| {
            if index < prepared.map.h_offset {
                shards[index % n].1.bit(index / n)
            } else {
                p_h.bit(index - prepared.map.h_offset)
            }
        });
    }
    let words = p.rows() / 64;
    let instance_blocks = n / 64;
    let tile_words = 64 * instance_blocks;
    // A job owns whole columns and whole transpose tiles. In particular,
    // a tile spanning several columns is transposed once, not per column.
    let columns_per_job = tile_words.div_ceil(words);
    let job_words = columns_per_job * words;
    let sha_words = prepared.map.h_offset / 64;
    let fill = |job: usize| {
        let count = columns_per_job.min(p.cols() - job * columns_per_job);
        let mut columns = vec![vec![0u64; words]; count];
        let start = job * job_words;
        let end = start + columns.len() * words;
        let mut tile = [0u64; 64];
        for tile_start in (start..end.min(sha_words)).step_by(tile_words) {
            let lb = tile_start / tile_words;
            let locals = ((end.min(sha_words) - tile_start) / instance_blocks).min(64);
            for block in 0..instance_blocks {
                for (i, word) in tile.iter_mut().enumerate() {
                    *word = masked_word(&shards[64 * block + i].1, lb);
                }
                transpose64(&mut tile);
                for (local, word) in tile[..locals].iter().enumerate() {
                    let dst = tile_start - start + local * instance_blocks + block;
                    columns[dst / words][dst % words] = *word;
                }
            }
        }
        for dst in start.max(sha_words)..end.min(sha_words + p_h.words().len()) {
            columns[(dst - start) / words][(dst - start) % words] =
                masked_word(p_h, dst - sha_words);
        }
        columns
    };
    let jobs = p.cols().div_ceil(columns_per_job);
    #[cfg(feature = "parallel")]
    let groups: Vec<_> = (0..jobs).into_par_iter().map(fill).collect();
    #[cfg(not(feature = "parallel"))]
    let groups: Vec<_> = (0..jobs).map(fill).collect();
    groups.into_iter().flatten().collect()
}

/// Computes the SHA trace and all hint values. Signing is not part of this API.
pub fn generate_sha256_ecdsa_witness(
    prepared: &PreparedSha256Ecdsa,
    statement: &Sha256EcdsaStatement,
    message: &[u8],
) -> Result<Sha256EcdsaWitness> {
    if statement.log_compressions as usize != prepared.log_n
        || message.len() != prepared.message_bytes()
    {
        return Err(error(
            "message length or compression count differs from the prepared relation",
        ));
    }
    let blocks: Vec<[u32; 16]> = message
        .chunks_exact(64)
        .map(|bytes| {
            array::from_fn(|w| u32::from_be_bytes(bytes[w * 4..w * 4 + 4].try_into().unwrap()))
        })
        .chain(std::iter::once(prepared.padding()))
        .collect();
    let mut states = Vec::with_capacity(blocks.len() + 1);
    states.push(sha256::INITIAL_STATE);
    for block in &blocks {
        states.push(super::super::sha256::sha256_compress(
            *states.last().unwrap(),
            *block,
        ));
    }
    let generate = |i: usize| -> Result<(PackedWitness, PackedWitness)> {
        let input: [bool; sha256::COMPRESSION_INPUT_BITS] = array::from_fn(|bit| {
            if bit < 512 {
                blocks[i][bit / 32] >> (bit % 32) & 1 != 0
            } else {
                states[i][(bit - 512) / 32] >> (bit % 32) & 1 != 0
            }
        });
        let mut generator = Witgen::with_inputs_and_capacity(
            &input,
            sha256::COMPRESSION_INPUT_BITS + sha256::COMPRESSION_HINT_BITS,
        );
        let output = sha256::compression_circuit(&mut generator, &input);
        for (bit, value) in output.into_iter().enumerate() {
            if value != (states[i + 1][bit / 32] >> (bit % 32) & 1 != 0) {
                return Err(error("SHA circuit/native digest mismatch"));
            }
        }
        let pair = generator.into_witnesses();
        if pair.1.bit_len() != SHA_H {
            return Err(error("unexpected SHA assignment width"));
        }
        Ok(pair)
    };
    #[cfg(feature = "parallel")]
    let shards: Vec<_> = (0..blocks.len())
        .into_par_iter()
        .map(generate)
        .collect::<Result<_>>()?;
    #[cfg(not(feature = "parallel"))]
    let shards: Vec<_> = (0..blocks.len()).map(generate).collect::<Result<_>>()?;

    let digest: [u8; 32] =
        array::from_fn(|byte| states.last().unwrap()[byte / 4].to_be_bytes()[byte % 4]);
    let rinv = inverse(&statement.r)?;
    let sinv = inverse(&statement.s)?;
    let words = [
        &digest,
        &statement.qx,
        &statement.qy,
        &statement.r,
        &statement.s,
        &rinv,
        &sinv,
    ];
    let input: [bool; p256::VERIFY_DIGEST_INPUT_BITS] =
        array::from_fn(|bit| words[bit / 256][31 - (bit % 256) / 8] >> (bit % 8) & 1 != 0);
    let mut generator =
        ProductWitgen::with_inputs_and_capacity(&input, p256::VERIFY_DIGEST_WITNESS_BITS);
    p256::verify_digest_circuit(&mut generator, &input);
    let (p_f, p_h, products) = generator.into_parts();
    if p_h.bit_len() != prepared.local.p_map.rows()
        || p_f.bit_len() + 1 != prepared.local.p_map.cols()
    {
        return Err(error("unexpected P-256 witness width"));
    }
    let f_rows = pack_source(prepared, &shards, &p_f);
    let h_rows = pack_assignment(prepared, &shards, &p_h);
    Ok(Sha256EcdsaWitness {
        f_rows: f_rows.into(),
        h_rows,
        products,
        statement: statement.clone(),
    })
}

#[cfg(test)]
mod packing_tests {
    use super::copy_bits;

    #[test]
    fn column_copies_match_bitwise_oracle_across_boundaries() {
        let source = [0x0102_0304_0506_0708u64, u64::MAX, 0xAA55_AA55_AA55_AA55, 0];
        for dst in [0, 1, 63, 64, 127, 128, 129] {
            for src in [0, 1, 63] {
                for len in [0, 1, 63, 64, 127, 129, 190] {
                    let mut rows = vec![vec![0u64; 2]; 4];
                    super::copy_column_bits(&mut rows, 7, dst, &source, src, len);
                    let actual: Vec<_> = rows.into_iter().flatten().collect();
                    let mut expected = vec![0u64; 8];
                    for bit in 0..len {
                        let value = source[(src + bit) / 64] >> ((src + bit) % 64) & 1;
                        expected[(dst + bit) / 64] |= value << ((dst + bit) % 64);
                    }
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn unaligned_copies_match_individual_bits() {
        copy_bits(&mut [], 0, &[], 0, 0);
        for pattern in 0..3 {
            let source: Vec<u64> = (0..20)
                .map(|i| match pattern {
                    0 => 0,
                    1 => u64::MAX,
                    _ => (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                })
                .collect();
            for src_off in 64usize..128 {
                for dst_off in 0usize..64 {
                    for len in [0, 1, 2, 31, 63, 64, 65, 127, 128, 129, 191, 1024] {
                        // Preserve nonzero bits outside the copy and provide
                        // exactly the source storage needed by this range.
                        let mut actual = vec![0xa5a5_a5a5_a5a5_a5a5; 20];
                        for bit in dst_off..dst_off + len {
                            actual[bit / 64] &= !(1u64 << (bit % 64));
                        }
                        let mut expected = actual.clone();
                        for bit in 0..len {
                            let value =
                                (source[(src_off + bit) / 64] >> ((src_off + bit) % 64)) & 1;
                            expected[(dst_off + bit) / 64] |= value << ((dst_off + bit) % 64);
                        }
                        copy_bits(
                            &mut actual,
                            dst_off,
                            &source[..(src_off + len).div_ceil(64)],
                            src_off,
                            len,
                        );
                        assert_eq!(
                            actual, expected,
                            "source {src_off}, destination {dst_off}, length {len}"
                        );
                    }
                }
            }
        }
    }
}
