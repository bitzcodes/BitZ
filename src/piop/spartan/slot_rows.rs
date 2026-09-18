//! Slot-major bit-row packing shared by the compact Spartan witnesses.
//!
//! A slot-major witness commits 128 bit slots per gate. With `s` clear
//! column coordinates and `h` high gate coordinates
//! (`high_gate_count = 2^h`), BitZ row `c` holds the slots of every gate
//! `(gate_high << s) | c` as `128 / W` lanes of `high_gate_count` `W`-bit
//! cells: bit `j` of the cell at lane `word_slot`, position `gate_high` is
//! slot `word_slot * W + j` of that gate — the layout of
//! `BabyBearMulLayout::bitz_cell` and `MulLayout::<u32>::bitz_bit_position`.
//!
//! Writing the rows one bit at a time scatters `128 · gates`
//! read-modify-writes across the rows (one cache line per bit). Here a
//! block of `64 / W` consecutive `gate_high`s of one column is one bit
//! (`W = 1`) or byte (`W = 8`) transpose of the gates' packed slot words
//! whose output words land directly in the row. Tasks own groups of
//! adjacent columns, so the per-gate value arrays are read one cache line
//! at a time.

use crate::ligerito::transpose_64x64;
#[cfg(test)]
use crate::piop::spartan::mul::MulLayout;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Adjacent columns per parallel task: one cache line of `u64` values.
const COLUMNS_PER_TASK: usize = 8;
const WORD_BITS: usize = u64::BITS as usize;

/// Packs `W = 1` rows: 128 lanes of `high_gate_count` bits per column.
///
/// `gate_slots(gate)` returns the gate's slots as `(slots 0..64,
/// slots 64..128)`; gates at or beyond `live_gates` are all-zero and never
/// queried. Every row must hold `2 · high_gate_count` words, and
/// `high_gate_count` must be a multiple of 64 so each lane spans whole
/// words.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn pack_slot_major_rows_w1<F>(
    rows: &mut [Vec<u64>],
    s: usize,
    high_gate_count: usize,
    live_gates: usize,
    gate_slots: F,
) where
    F: Fn(usize) -> (u64, u64) + Sync,
{
    assert!(
        high_gate_count.is_multiple_of(WORD_BITS),
        "W=1 slot lanes must span whole words"
    );
    assert!(
        rows.iter().all(|row| row.len() == 2 * high_gate_count),
        "each W=1 row holds 128 lanes of high_gate_count bits"
    );
    let lane_words = high_gate_count / WORD_BITS;

    crate::cfg_chunks_mut!(rows, COLUMNS_PER_TASK)
        .enumerate()
        .for_each(|(task, group)| {
            let first_column = task * COLUMNS_PER_TASK;
            let mut lo = [[0_u64; WORD_BITS]; COLUMNS_PER_TASK];
            let mut hi = [[0_u64; WORD_BITS]; COLUMNS_PER_TASK];
            for block in 0..lane_words {
                for g in 0..WORD_BITS {
                    let gate_base = ((block * WORD_BITS + g) << s) | first_column;
                    for (i, (lo_i, hi_i)) in lo
                        .iter_mut()
                        .zip(hi.iter_mut())
                        .take(group.len())
                        .enumerate()
                    {
                        let gate = gate_base + i;
                        let (lo_word, hi_word) = if gate < live_gates {
                            gate_slots(gate)
                        } else {
                            (0, 0)
                        };
                        lo_i[g] = lo_word;
                        hi_i[g] = hi_word;
                    }
                }
                for ((row, lo_i), hi_i) in group.iter_mut().zip(lo.iter_mut()).zip(hi.iter_mut()) {
                    transpose_64x64(lo_i);
                    transpose_64x64(hi_i);
                    for (slot, word) in lo_i.iter().enumerate() {
                        row[slot * lane_words + block] = *word;
                    }
                    for (slot, word) in hi_i.iter().enumerate() {
                        row[(WORD_BITS + slot) * lane_words + block] = *word;
                    }
                }
            }
        });
}

/// Packs four native u32 limbs, fusing the first transpose stage into the
/// loads. LANES is 32 for bit cells or 4 for byte cells; each output word
/// contains two LANES-sized groups of gates.
pub(crate) fn pack_slot_major_u32<const LANES: usize, const W: usize, const COLUMNS: usize>(
    rows: &mut [Vec<u64>],
    s: usize,
    high_gate_count: usize,
    live_gates: usize,
    limbs: [&[u32]; 4],
) {
    assert!(LANES * W == 32 && LANES.is_power_of_two());
    assert!(high_gate_count.is_multiple_of(2 * LANES));
    assert!(rows.iter().all(|row| row.len() == 2 * high_gate_count));
    let limbs = limbs.map(|v| &v[..live_gates]);
    let lane_words = high_gate_count / (2 * LANES);
    crate::cfg_chunks_mut!(rows, COLUMNS)
        .enumerate()
        .for_each(|(task, group)| {
            let first_column = task * COLUMNS;
            let mut blocks = [[[0u64; LANES]; 4]; COLUMNS];
            for block in 0..lane_words {
                for g in 0..LANES {
                    let gate_base = ((block * 2 * LANES + g) << s) | first_column;
                    for (i, column) in blocks.iter_mut().take(group.len()).enumerate() {
                        let a = gate_base + i;
                        let b = a + (LANES << s);
                        for (matrix, limb) in column.iter_mut().zip(limbs) {
                            let lo = if a < live_gates { limb[a] } else { 0 };
                            let hi = if b < live_gates { limb[b] } else { 0 };
                            matrix[g] = u64::from(lo) | (u64::from(hi) << 32);
                        }
                    }
                }
                for (row, column) in group.iter_mut().zip(blocks.iter_mut()) {
                    for (limb, matrix) in column.iter_mut().enumerate() {
                        transpose_lanes::<LANES, W>(matrix);
                        for (lane, &word) in matrix.iter().enumerate() {
                            row[(limb * LANES + lane) * lane_words + block] = word;
                        }
                    }
                }
            }
        });
}

/// Swap stages shared by the complete byte transpose and the native u32
/// transposes whose first stage is already fused into their loads.
fn transpose_lanes<const N: usize, const W: usize>(m: &mut [u64; N]) {
    let mut j = N / 2;
    let mut shift = j * W;
    let mut mask = u64::MAX / ((1u64 << shift) + 1);
    while j != 0 {
        let mut k = 0;
        while k < N {
            let t = ((m[k] >> shift) ^ m[k | j]) & mask;
            m[k | j] ^= t;
            m[k] ^= t << shift;
            k = ((k | j) + 1) & !j;
        }
        j >>= 1;
        shift >>= 1;
        mask ^= mask << shift;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mix(gate: usize, salt: u64) -> u64 {
        let mut z = (gate as u64)
            .wrapping_add(salt)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z ^= z >> 29;
        z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z ^ (z >> 32)
    }

    fn slot_words(gate: usize) -> (u64, u64) {
        (mix(gate, 0x1111), mix(gate, 0x2222))
    }

    /// One read-modify-write per set slot bit, straight from the layout
    /// definition in the module docs.
    fn reference_rows(
        s: usize,
        high_gate_count: usize,
        live_gates: usize,
        word_bits: usize,
    ) -> Vec<Vec<u64>> {
        let h = high_gate_count.trailing_zeros() as usize;
        let cols = 1_usize << s;
        let mut rows = vec![vec![0_u64; 2 * high_gate_count]; cols];
        for gate in 0..live_gates {
            let (lo, hi) = slot_words(gate);
            let column = gate & (cols - 1);
            let gate_high = gate >> s;
            for slot in 0..128 {
                let bit = if slot < 64 {
                    (lo >> slot) & 1
                } else {
                    (hi >> (slot - 64)) & 1
                };
                if bit == 0 {
                    continue;
                }
                let b = ((slot / word_bits) << h) | gate_high;
                let packed_bit = b * word_bits + slot % word_bits;
                rows[column][packed_bit / 64] |= 1 << (packed_bit % 64);
            }
        }
        rows
    }

    #[test]
    fn w1_packing_matches_the_bitwise_reference() {
        // Lanes of 64..256 gates, column counts below and above one task,
        // and live counts that leave padded gates and partial blocks.
        for (s, h, live) in [
            (3, 6, 512),
            (4, 6, 700),
            (4, 6, 1),
            (5, 7, 4096),
            (8, 8, (1 << 16) - 5),
        ] {
            let high_gate_count = 1_usize << h;
            let mut rows = vec![vec![0_u64; 2 * high_gate_count]; 1 << s];
            pack_slot_major_rows_w1(&mut rows, s, high_gate_count, live, slot_words);
            assert_eq!(
                rows,
                reference_rows(s, high_gate_count, live, 1),
                "s={s} h={h} live={live}"
            );
        }
    }

    #[test]
    fn w8_packing_matches_the_bitwise_reference() {
        for (s, h, live) in [
            (3, 3, 60),
            (4, 4, 256),
            (4, 6, 700),
            (4, 6, 1),
            (5, 7, 4096),
            (8, 8, (1 << 16) - 5),
        ] {
            let high_gate_count = 1_usize << h;
            let mut rows = vec![vec![0_u64; 2 * high_gate_count]; 1 << s];
            let limbs: [Vec<u32>; 4] = core::array::from_fn(|limb| {
                (0..live)
                    .map(|gate| {
                        let (lo, hi) = slot_words(gate);
                        let word = if limb < 2 { lo } else { hi };
                        (word >> (32 * (limb % 2))) as u32
                    })
                    .collect()
            });
            pack_slot_major_u32::<4, 8, 32>(
                &mut rows, s, high_gate_count, live, limbs.each_ref().map(Vec::as_slice),
            );
            assert_eq!(
                rows,
                reference_rows(s, high_gate_count, live, 8),
                "s={s} h={h} live={live}"
            );
        }
    }

    #[test]
    fn byte_transpose_matches_the_naive_transpose_and_is_an_involution() {
        let input: [u64; 8] = core::array::from_fn(|k| mix(k, 0x3333));
        let mut expected = [0_u64; 8];
        for (i, out) in expected.iter_mut().enumerate() {
            for (k, word) in input.iter().enumerate() {
                *out |= ((word >> (8 * i)) & 0xFF) << (8 * k);
            }
        }
        let mut m = input;
        transpose_lanes::<8, 8>(&mut m);
        assert_eq!(m, expected);
        transpose_lanes::<8, 8>(&mut m);
        assert_eq!(m, input);
    }
}

/// Packs `W = 1` rows for gates with `64 · N` bit slots: `64 · N` lanes of
/// `high_gate_count` bits per column.
///
/// `gate_slots(gate)` returns the gate's slots as `N` little-endian words
/// (word `w` holds slots `64w..64w+64`); gates at or beyond `live_gates` are
/// all-zero and never queried. Every row must hold `N · high_gate_count`
/// words, and `high_gate_count` must be a multiple of 64 so each lane spans
/// whole words. Same transpose scheme as [`pack_slot_major_rows_w1`], which
/// stays the 128-slot packer.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn pack_slot_major_rows_w1_words<const N: usize, F>(
    rows: &mut [Vec<u64>],
    s: usize,
    high_gate_count: usize,
    live_gates: usize,
    gate_slots: F,
) where
    F: Fn(usize) -> [u64; N] + Sync,
{
    assert!(
        high_gate_count.is_multiple_of(WORD_BITS),
        "W=1 slot lanes must span whole words"
    );
    assert!(
        rows.iter().all(|row| row.len() == N * high_gate_count),
        "each W=1 row holds 64·N lanes of high_gate_count bits"
    );
    let lane_words = high_gate_count / WORD_BITS;

    crate::cfg_chunks_mut!(rows, COLUMNS_PER_TASK)
        .enumerate()
        .for_each(|(task, group)| {
            let first_column = task * COLUMNS_PER_TASK;
            let mut blocks = vec![[[0_u64; WORD_BITS]; N]; COLUMNS_PER_TASK];
            for block in 0..lane_words {
                for g in 0..WORD_BITS {
                    let gate_base = ((block * WORD_BITS + g) << s) | first_column;
                    for (i, column_blocks) in blocks.iter_mut().take(group.len()).enumerate() {
                        let gate = gate_base + i;
                        let words = if gate < live_gates {
                            gate_slots(gate)
                        } else {
                            [0; N]
                        };
                        for (word, target) in words.iter().zip(column_blocks.iter_mut()) {
                            target[g] = *word;
                        }
                    }
                }
                for (row, column_blocks) in group.iter_mut().zip(blocks.iter_mut()) {
                    for (word_index, matrix) in column_blocks.iter_mut().enumerate() {
                        transpose_64x64(matrix);
                        for (slot, word) in matrix.iter().enumerate() {
                            row[(word_index * WORD_BITS + slot) * lane_words + block] = *word;
                        }
                    }
                }
            }
        });
}
