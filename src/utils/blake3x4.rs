//! Batched BLAKE3 single-block compression for proof-of-work nonce scans.
//!
//! Every BitZ grind tests `blake3(prefix ‖ nonce_le)` for consecutive nonces
//! — the Ligerito challenger's 16-byte seed, the Spartan/forest/Round-0
//! boundaries' 32-byte seed — one message of at most 64 bytes per attempt,
//! so one compression each. The `blake3` crate runs single compressions
//! through its portable scalar kernel on aarch64 (its NEON code is the
//! multi-chunk `hash_many` path), which made the level-0 fold grinding of a
//! Johnson-regime opener (2^21–2^23 attempts at the hybrid shapes) the
//! largest part of its Ligerito time. [`first_pow_nonce`] compresses eight
//! nonces at once in NEON or AVX2 lanes and returns exactly the nonce a serial scan
//! would: the SMALLEST hit of the range. Every kernel constant is the BLAKE3
//! specification's (the equivalence tests below pin the lanes to
//! `blake3::hash` bit for bit).

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];
/// One message of at most 64 bytes: one block of one chunk, hashed as
/// the root.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const FLAGS: u32 = 1 | 2 | 8; // CHUNK_START | CHUNK_END | ROOT

#[cfg(target_arch = "x86_64")]
mod avx2;

/// Longest prefix (bytes) that keeps `prefix ‖ nonce` inside one block.
pub(crate) const MAX_PREFIX_LEN: usize = 56;

/// `blake3(prefix ‖ nonce_le)` has at least `bits` leading zero bits.
pub(crate) fn pow_ok(prefix: &[u8], nonce: u64, bits: u32) -> bool {
    assert!(
        prefix.len() <= MAX_PREFIX_LEN,
        "proof-of-work prefix exceeds one block"
    );
    let mut buf = [0u8; 64];
    buf[..prefix.len()].copy_from_slice(prefix);
    buf[prefix.len()..prefix.len() + 8].copy_from_slice(&nonce.to_le_bytes());
    let hash = blake3::hash(&buf[..prefix.len() + 8]);
    leading_zero_bits(hash.as_bytes()) >= bits
}

pub(crate) fn leading_zero_bits(bytes: &[u8]) -> u32 {
    let mut acc = 0u32;
    for &b in bytes {
        if b == 0 {
            acc = acc.wrapping_add(8);
        } else {
            return acc.wrapping_add(b.leading_zeros());
        }
    }
    acc
}

/// The smallest nonce in `start..end` whose `blake3(prefix ‖ nonce_le)`
/// has at least `bits` leading zero bits, scanning in increasing order.
/// `prefix` is at most [`MAX_PREFIX_LEN`] bytes; the NEON lanes take a
/// prefix of whole 32-bit words (every BitZ seed), as do the AVX2 lanes; other inputs scan
/// through the reference hash.
pub(crate) fn first_pow_nonce(prefix: &[u8], start: u64, end: u64, bits: u32) -> Option<u64> {
    assert!(
        prefix.len() <= MAX_PREFIX_LEN,
        "proof-of-work prefix exceeds one block"
    );
    #[cfg(target_arch = "aarch64")]
    if prefix.len() % 4 == 0 {
        // SAFETY: NEON is a baseline feature of every aarch64 target.
        return unsafe { neon::first_pow_nonce(prefix, start, end, bits) };
    }
    #[cfg(target_arch = "x86_64")]
    if prefix.len() % 4 == 0 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support was checked above; the public prefix bound is
        // asserted at entry and the kernel accepts whole-word prefixes.
        return unsafe { avx2::first_pow_nonce(prefix, start, end, bits) };
    }
    (start..end).find(|&n| pow_ok(prefix, n, bits))
}

/// The smallest nonce whose `blake3(prefix ‖ nonce_le)` has at least `bits`
/// leading zero bits, scanned by every thread of the pool: threads take
/// consecutive chunks of the nonce space from a shared counter and keep the
/// running minimum hit; a thread stops once its next chunk starts at or
/// above that minimum. Every chunk below the smallest hit's chunk is taken
/// (the counter is monotonic) and scanned to completion, each chunk yields
/// its own smallest hit, so the result is exactly the serial scan's nonce
/// whatever the scheduling — a single thread running every task included.
/// Unlike a wave-synchronised search, no chunk beyond the hit's is scanned
/// except those already in flight. `None` once the `u64` nonce space is
/// exhausted (a theoretical bound only).
#[cfg(feature = "parallel")]
pub(crate) fn smallest_pow_nonce(prefix: &[u8], bits: u32) -> Option<u64> {
    use std::sync::atomic::{AtomicU64, Ordering};
    const CHUNK_LOG: u64 = 10;
    let next = AtomicU64::new(0);
    let best = AtomicU64::new(u64::MAX);
    rayon::broadcast(|_| {
        loop {
            let chunk = next.fetch_add(1, Ordering::Relaxed);
            if chunk >= 1 << (64 - CHUNK_LOG) {
                break;
            }
            let start = chunk << CHUNK_LOG;
            if start >= best.load(Ordering::Relaxed) {
                break;
            }
            if let Some(n) =
                first_pow_nonce(prefix, start, start.saturating_add(1 << CHUNK_LOG), bits)
            {
                best.fetch_min(n, Ordering::Relaxed);
            }
        }
    });
    let found = best.load(Ordering::Relaxed);
    // The last nonce of the space is excluded by the half-open chunks.
    if found == u64::MAX && !pow_ok(prefix, u64::MAX, bits) {
        return None;
    }
    Some(found)
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    use super::{FLAGS, IV, MSG_SCHEDULE};

    /// Byte shuffle rotating every 32-bit lane right by 8 bits.
    const ROTR8_BYTES: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];

    #[inline(always)]
    unsafe fn rotr16(x: uint32x4_t) -> uint32x4_t {
        vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x)))
    }
    #[inline(always)]
    unsafe fn rotr12(x: uint32x4_t) -> uint32x4_t {
        vsriq_n_u32::<12>(vshlq_n_u32::<20>(x), x)
    }
    #[inline(always)]
    unsafe fn rotr8(x: uint32x4_t, idx: uint8x16_t) -> uint32x4_t {
        vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(x), idx))
    }
    #[inline(always)]
    unsafe fn rotr7(x: uint32x4_t) -> uint32x4_t {
        vsriq_n_u32::<7>(vshlq_n_u32::<25>(x), x)
    }

    /// Independent four-lane compressions interleaved per instruction, so
    /// their dependency chains overlap (one G is an 8-deep chain).
    const GROUPS: usize = 2;
    const NONCES_PER_CALL: u64 = 4 * GROUPS as u64;

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn g(
        v: &mut [[uint32x4_t; 16]; GROUPS],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        x: [uint32x4_t; GROUPS],
        y: [uint32x4_t; GROUPS],
        idx: uint8x16_t,
    ) {
        for (v, (x, y)) in v.iter_mut().zip(x.into_iter().zip(y)) {
            v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), x);
            v[d] = rotr16(veorq_u32(v[d], v[a]));
            v[c] = vaddq_u32(v[c], v[d]);
            v[b] = rotr12(veorq_u32(v[b], v[c]));
            v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), y);
            v[d] = rotr8(veorq_u32(v[d], v[a]), idx);
            v[c] = vaddq_u32(v[c], v[d]);
            v[b] = rotr7(veorq_u32(v[b], v[c]));
        }
    }

    #[inline(always)]
    unsafe fn round(
        v: &mut [[uint32x4_t; 16]; GROUPS],
        m: &[[uint32x4_t; 16]; GROUPS],
        r: usize,
        idx: uint8x16_t,
    ) {
        let s = &MSG_SCHEDULE[r];
        let w = |i: usize| -> [uint32x4_t; GROUPS] { core::array::from_fn(|k| m[k][s[i]]) };
        g(v, 0, 4, 8, 12, w(0), w(1), idx);
        g(v, 1, 5, 9, 13, w(2), w(3), idx);
        g(v, 2, 6, 10, 14, w(4), w(5), idx);
        g(v, 3, 7, 11, 15, w(6), w(7), idx);
        g(v, 0, 5, 10, 15, w(8), w(9), idx);
        g(v, 1, 6, 11, 12, w(10), w(11), idx);
        g(v, 2, 7, 8, 13, w(12), w(13), idx);
        g(v, 3, 4, 9, 14, w(14), w(15), idx);
    }

    /// First output word (`state[0] ^ state[8]`, the hash's bytes 0..4 in
    /// little-endian order) of `blake3(prefix ‖ nonce_le)` for the nonces
    /// `base..base + NONCES_PER_CALL`, one per lane, in nonce order. The
    /// prefix is `prefix.len()` whole words (at most 14).
    #[inline(always)]
    unsafe fn first_words(prefix: &[u32], base: u64) -> [u32; NONCES_PER_CALL as usize] {
        let idx = vld1q_u8(ROTR8_BYTES.as_ptr());
        let zero = vdupq_n_u32(0);
        let words = prefix.len();
        let block_len = 4 * (words as u32 + 2);
        let mut m = [[zero; 16]; GROUPS];
        let mut v = [[zero; 16]; GROUPS];
        for (k, (m, v)) in m.iter_mut().zip(v.iter_mut()).enumerate() {
            let first = base.wrapping_add(4 * k as u64);
            let lo: [u32; 4] = core::array::from_fn(|j| first.wrapping_add(j as u64) as u32);
            let hi: [u32; 4] =
                core::array::from_fn(|j| (first.wrapping_add(j as u64) >> 32) as u32);
            for (slot, &w) in m.iter_mut().zip(prefix) {
                *slot = vdupq_n_u32(w);
            }
            m[words] = vld1q_u32(lo.as_ptr());
            m[words + 1] = vld1q_u32(hi.as_ptr());
            for (slot, &w) in v.iter_mut().zip(IV.iter().chain(&IV[..4])) {
                *slot = vdupq_n_u32(w);
            }
            v[14] = vdupq_n_u32(block_len);
            v[15] = vdupq_n_u32(FLAGS);
        }
        for r in 0..7 {
            round(&mut v, &m, r, idx);
        }
        let mut out = [0u32; NONCES_PER_CALL as usize];
        for (k, v) in v.iter().enumerate() {
            vst1q_u32(out.as_mut_ptr().add(4 * k), veorq_u32(v[0], v[8]));
        }
        out
    }

    pub(super) unsafe fn first_pow_nonce(
        prefix: &[u8],
        start: u64,
        end: u64,
        bits: u32,
    ) -> Option<u64> {
        debug_assert!(prefix.len() % 4 == 0 && prefix.len() <= super::MAX_PREFIX_LEN);
        let words: Vec<u32> = prefix
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        let mut base = start;
        while base < end {
            let w0 = first_words(&words, base);
            for (k, &w) in w0.iter().enumerate() {
                let nonce = base.wrapping_add(k as u64);
                if nonce >= end {
                    return None;
                }
                // Leading zero bits of the hash's byte string: bytes 0..4
                // are this word little-endian, so byte-swap and count.
                let leading = w.swap_bytes().leading_zeros();
                if leading >= bits.min(32) && (bits <= 32 || super::pow_ok(prefix, nonce, bits)) {
                    return Some(nonce);
                }
            }
            base = base.wrapping_add(NONCES_PER_CALL);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pseudo-random prefixes of every length the grinders use (16 and 32
    /// bytes) plus the edges of the single-block kernel and a length that
    /// takes the scalar path.
    fn prefix(i: u64, len: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for k in 0..2u64 {
            out.extend_from_slice(blake3::hash(&(i * 2 + k).to_le_bytes()).as_bytes());
        }
        out.truncate(len);
        out
    }
    const LENGTHS: [usize; 6] = [16, 32, 0, 4, 56, 7];

    /// Compare every bit of each output word, not just its leading-zero count.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_words_match_the_blake3_crate() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for seed in 0..8 {
            for len in (0..=MAX_PREFIX_LEN).step_by(4) {
                let prefix = prefix(seed, len);
                let words: Vec<_> = prefix
                    .chunks_exact(4)
                    .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                    .collect();
                for base in [0, 1, 4093, (1 << 32) - 3, u64::MAX - 7] {
                    // SAFETY: AVX2 is available; all prefixes and full batches
                    // satisfy the kernel's documented bounds.
                    let actual = unsafe { avx2::first_words(&words, base) };
                    for (lane, word) in actual.into_iter().enumerate() {
                        let mut message = prefix.clone();
                        message.extend_from_slice(&(base + lane as u64).to_le_bytes());
                        let hash = blake3::hash(&message);
                        let expected = u32::from_le_bytes(hash.as_bytes()[..4].try_into().unwrap());
                        assert_eq!(
                            word, expected,
                            "seed {seed} len {len} base {base} lane {lane}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn range_edges_match_the_serial_scan() {
        for len in 0..=MAX_PREFIX_LEN {
            let prefix = prefix(700 + len as u64, len);
            for start in [0, (1 << 32) - 9, u64::MAX - 17] {
                for count in [0, 1, 7, 8, 9, 15, 16, 17] {
                    let end = start + count;
                    for bits in [0, 1, 5, 31, 32, 33, 256, 257] {
                        let expected = (start..end).find(|&nonce| pow_ok(&prefix, nonce, bits));
                        assert_eq!(
                            first_pow_nonce(&prefix, start, end, bits),
                            expected,
                            "len {len} start {start} count {count} bits {bits}"
                        );
                    }
                }
            }
            assert_eq!(first_pow_nonce(&prefix, 17, 4, 0), None);
            assert_eq!(first_pow_nonce(&prefix, u64::MAX, u64::MAX, 0), None);
        }
    }

    /// The lanes reproduce `blake3::hash` bit for bit (the first word is
    /// what the scan decides on), including across a 2^32 nonce boundary.
    #[test]
    fn lanes_match_the_blake3_crate() {
        for s in 0..6u64 {
            for len in LENGTHS {
                let seed = prefix(s, len);
                for start in [0u64, 1, 5, 4093, (1 << 32) - 3, u64::MAX - 9] {
                    for n in start..start.saturating_add(40) {
                        let mut buf = seed.clone();
                        buf.extend_from_slice(&n.to_le_bytes());
                        let reference = blake3::hash(&buf);
                        let bytes = reference.as_bytes();
                        // A range that forces the scan to test exactly this
                        // nonce with a threshold it meets: `bits` = its own
                        // leading-zero count (capped at 32 for the lane check).
                        let lz = leading_zero_bits(&bytes[..4]).min(32);
                        assert_eq!(
                            first_pow_nonce(&seed, n, n + 1, lz),
                            Some(n),
                            "len {len} seed {s} nonce {n}"
                        );
                        if lz < 32 {
                            assert_eq!(
                                first_pow_nonce(&seed, n, n + 1, lz + 1),
                                None,
                                "len {len} seed {s} nonce {n}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Range scans return the serial scan's smallest hit, or none.
    #[test]
    fn range_scans_match_the_serial_scan() {
        for s in 0..4u64 {
            for len in LENGTHS {
                let seed = prefix(s, len);
                for bits in [0u32, 1, 3, 6, 9, 12] {
                    for (start, n) in [
                        (0u64, 1u64),
                        (0, 3),
                        (7, 4097),
                        (1 << 20, 5000),
                        ((1 << 32) - 5, 1000),
                    ] {
                        let expected = (start..start + n).find(|&x| pow_ok(&seed, x, bits));
                        assert_eq!(
                            first_pow_nonce(&seed, start, start + n, bits),
                            expected,
                            "len {len} seed {s} bits {bits} start {start}"
                        );
                    }
                }
            }
        }
    }

    /// The pool-wide search returns the serial minimum for both seed widths.
    #[cfg(feature = "parallel")]
    #[test]
    fn pool_search_returns_the_serial_minimum() {
        for s in 0..3u64 {
            for len in [16usize, 32] {
                let seed = prefix(s + 40, len);
                for bits in [12u32, 14] {
                    let serial = (0..u64::MAX).find(|&x| pow_ok(&seed, x, bits));
                    assert_eq!(
                        smallest_pow_nonce(&seed, bits),
                        serial,
                        "len {len} seed {s} bits {bits}"
                    );
                }
            }
        }
    }

    /// Thresholds above 32 bits fall back to the full hash comparison.
    #[test]
    fn wide_thresholds_use_the_full_hash() {
        for len in [16usize, 32] {
            let seed = prefix(99, len);
            // No short input has 40 leading zero bits with noticeable
            // probability: a short range must report none, exactly like the
            // serial scan.
            assert_eq!(first_pow_nonce(&seed, 0, 20_000, 40), None);
            assert_eq!((0..20_000u64).find(|&n| pow_ok(&seed, n, 40)), None);
        }
    }
}
