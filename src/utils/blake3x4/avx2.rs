//! Eight independent single-block BLAKE3 compressions in AVX2 lanes.
//!
//! Entry points require AVX2 support. The caller checks that feature before
//! entering this module; helpers inline into the feature-enabled kernel.

use super::{FLAGS, IV, MSG_SCHEDULE};
use core::arch::x86_64::*;

const LANES: usize = 8;

#[inline(always)]
unsafe fn g(v: &mut [__m256i; 16], a: usize, b: usize, c: usize, d: usize, x: __m256i, y: __m256i) {
    // SAFETY: this helper is only called by the AVX2-enabled compression.
    unsafe {
        let rotate16 = _mm256_setr_epi8(
            2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11,
            8, 9, 14, 15, 12, 13,
        );
        let rotate8 = _mm256_setr_epi8(
            1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10,
            11, 8, 13, 14, 15, 12,
        );
        v[a] = _mm256_add_epi32(_mm256_add_epi32(v[a], v[b]), x);
        v[d] = _mm256_shuffle_epi8(_mm256_xor_si256(v[d], v[a]), rotate16);
        v[c] = _mm256_add_epi32(v[c], v[d]);
        let b_xor = _mm256_xor_si256(v[b], v[c]);
        v[b] = _mm256_or_si256(
            _mm256_srli_epi32::<12>(b_xor),
            _mm256_slli_epi32::<20>(b_xor),
        );
        v[a] = _mm256_add_epi32(_mm256_add_epi32(v[a], v[b]), y);
        v[d] = _mm256_shuffle_epi8(_mm256_xor_si256(v[d], v[a]), rotate8);
        v[c] = _mm256_add_epi32(v[c], v[d]);
        let b_xor = _mm256_xor_si256(v[b], v[c]);
        v[b] = _mm256_or_si256(
            _mm256_srli_epi32::<7>(b_xor),
            _mm256_slli_epi32::<25>(b_xor),
        );
    }
}

/// First hash words in nonce order. The caller supplies at most 14 prefix
/// words and a base whose eight nonces fit in u64.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn first_words(prefix: &[u32], base: u64) -> [u32; LANES] {
    let zero = _mm256_setzero_si256();
    let mut message = [zero; 16];
    let mut state = [zero; 16];
    for (slot, &word) in message.iter_mut().zip(prefix) {
        *slot = _mm256_set1_epi32(word as i32);
    }
    let low: [u32; LANES] = core::array::from_fn(|lane| (base + lane as u64) as u32);
    let high: [u32; LANES] = core::array::from_fn(|lane| ((base + lane as u64) >> 32) as u32);
    // SAFETY: both arrays have exactly 32 readable bytes; the unaligned loads
    // impose no alignment requirement. Prefix bounds leave two nonce slots.
    unsafe {
        message[prefix.len()] = _mm256_loadu_si256(low.as_ptr().cast());
        message[prefix.len() + 1] = _mm256_loadu_si256(high.as_ptr().cast());
    }
    for (slot, &word) in state.iter_mut().zip(IV.iter().chain(&IV[..4])) {
        *slot = _mm256_set1_epi32(word as i32);
    }
    state[14] = _mm256_set1_epi32((4 * (prefix.len() + 2)) as i32);
    state[15] = _mm256_set1_epi32(FLAGS as i32);
    for schedule in &MSG_SCHEDULE {
        let word = |i: usize| message[schedule[i]];
        // SAFETY: this function enables AVX2; all state indexes are fixed.
        unsafe {
            g(&mut state, 0, 4, 8, 12, word(0), word(1));
            g(&mut state, 1, 5, 9, 13, word(2), word(3));
            g(&mut state, 2, 6, 10, 14, word(4), word(5));
            g(&mut state, 3, 7, 11, 15, word(6), word(7));
            g(&mut state, 0, 5, 10, 15, word(8), word(9));
            g(&mut state, 1, 6, 11, 12, word(10), word(11));
            g(&mut state, 2, 7, 8, 13, word(12), word(13));
            g(&mut state, 3, 4, 9, 14, word(14), word(15));
        }
    }
    let mut output = [0u32; LANES];
    // SAFETY: output has exactly 32 writable bytes and needs no alignment.
    unsafe {
        _mm256_storeu_si256(
            output.as_mut_ptr().cast(),
            _mm256_xor_si256(state[0], state[8]),
        );
    }
    output
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn first_pow_nonce(
    prefix: &[u8],
    start: u64,
    end: u64,
    bits: u32,
) -> Option<u64> {
    debug_assert!(prefix.len() <= super::MAX_PREFIX_LEN && prefix.len() % 4 == 0);
    let mut words = [0u32; super::MAX_PREFIX_LEN / 4];
    for (word, bytes) in words.iter_mut().zip(prefix.chunks_exact(4)) {
        *word = u32::from_le_bytes(bytes.try_into().expect("four-byte prefix word"));
    }
    let words = &words[..prefix.len() / 4];
    let mut base = start;
    while end.saturating_sub(base) >= LANES as u64 {
        // SAFETY: AVX2 is enabled, the prefix is bounded, and a complete batch
        // fits below end. In particular, no nonce addition can overflow.
        let hashes = unsafe { first_words(words, base) };
        for (lane, hash) in hashes.into_iter().enumerate() {
            let nonce = base + lane as u64;
            if hash.swap_bytes().leading_zeros() >= bits.min(32)
                && (bits <= 32 || super::pow_ok(prefix, nonce, bits))
            {
                return Some(nonce);
            }
        }
        base += LANES as u64;
    }
    // Includes empty/reversed ranges and the final zero to seven nonces.
    (base..end).find(|&nonce| super::pow_ok(prefix, nonce, bits))
}
