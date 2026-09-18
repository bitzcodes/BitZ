//! Byte-exact BLAKE3 recomputation of a long digest stream that varies only
//! inside its first chunk.
//!
//! The prepared-statement digest hashes a stream whose only
//! configuration-dependent bytes are the canonical modulus encoding near the
//! front; everything after it is a fixed function of the constraint matrices.
//! BLAKE3 is a binary tree over 1024-byte chunks, so replacing bytes inside
//! chunk 0 invalidates only that leaf and its ancestors along the left spine.
//! [`SplicedDigestBuilder`] consumes the full stream once (with placeholder
//! bytes in the hole) and retains chunk 0 verbatim plus the chaining values of
//! the left spine's sibling subtrees — the complete doubling ranges
//! `[2^j, 2^(j+1))` of chunks and the root's right subtree. Each
//! [`SplicedStreamDigest::digest_with`] then recomputes the exact
//! [`blake3::Hasher`] digest of the spliced stream from one chunk compression
//! and `O(log n)` parent merges, instead of rehashing the whole stream.
//!
//! The correctness anchor is [`tests::matches_plain_hasher_across_boundaries`]:
//! equality with a plain `Hasher` over streams crossing every tree-shape
//! boundary class (sub-chunk, exact chunk, power-of-two chunks, ragged tails).

use std::ops::Range;

use blake3::hazmat::{
    ChainingValue, HasherExt, Mode, merge_subtrees_non_root, merge_subtrees_root,
};
use blake3::{CHUNK_LEN, Hasher};

/// Why a spliced digest could not be constructed or evaluated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SplicedDigestError {
    /// The hole must lie inside the first chunk and inside the stream.
    HoleOutOfBounds {
        hole: Range<usize>,
        stream_len: usize,
    },
    /// `digest_with` was called with bytes of the wrong length.
    HoleLengthMismatch { expected: usize, actual: usize },
}

/// One doubling range of chunks `[start, end)` being hashed as a subtree.
struct RangeHasher {
    hasher: Hasher,
    end_byte: usize,
    fed: bool,
}

/// Streaming builder for [`SplicedStreamDigest`]. Feed the complete stream
/// exactly once through [`Self::push`]; the bytes inside `hole` may be
/// placeholders (they are never hashed at build time).
pub(crate) struct SplicedDigestBuilder {
    hole: Range<usize>,
    first_chunk: Vec<u8>,
    position: usize,
    current: Option<RangeHasher>,
    completed: Vec<ChainingValue>,
}

impl SplicedDigestBuilder {
    pub(crate) fn new(hole: Range<usize>) -> Self {
        Self {
            hole,
            first_chunk: Vec::with_capacity(CHUNK_LEN),
            position: 0,
            current: None,
            completed: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            if self.position < CHUNK_LEN {
                let take = bytes.len().min(CHUNK_LEN - self.position);
                self.first_chunk.extend_from_slice(&bytes[..take]);
                self.position += take;
                bytes = &bytes[take..];
                continue;
            }

            // Roll into the doubling range containing `position`: after
            // chunk 0 the ranges are chunks [1,2), [2,4), [4,8), ...; each
            // starts at chunk 2^j and is a complete subtree of the final
            // tree, so it can be hashed independently at its input offset.
            let needs_new_range = match &self.current {
                Some(range) => self.position == range.end_byte,
                None => true,
            };
            if needs_new_range {
                if let Some(range) = self.current.take() {
                    debug_assert!(range.fed);
                    self.completed.push(range.hasher.finalize_non_root());
                }
                debug_assert_eq!(self.position % CHUNK_LEN, 0);
                let start_chunk = self.position / CHUNK_LEN;
                debug_assert!(start_chunk.is_power_of_two());
                let mut hasher = Hasher::new();
                hasher.set_input_offset(self.position as u64);
                self.current = Some(RangeHasher {
                    hasher,
                    end_byte: 2 * start_chunk * CHUNK_LEN,
                    fed: false,
                });
            }

            let range = self.current.as_mut().expect("range was just ensured");
            let take = bytes.len().min(range.end_byte - self.position);
            range.hasher.update(&bytes[..take]);
            range.fed = true;
            self.position += take;
            bytes = &bytes[take..];
        }
    }

    pub(crate) fn finish(self) -> Result<SplicedStreamDigest, SplicedDigestError> {
        if self.hole.end > self.position || self.hole.end > CHUNK_LEN {
            return Err(SplicedDigestError::HoleOutOfBounds {
                hole: self.hole,
                stream_len: self.position,
            });
        }

        match self.current {
            // The whole stream fits in one (root) chunk.
            None => Ok(SplicedStreamDigest::Short {
                hole: self.hole,
                stream: self.first_chunk,
            }),
            Some(range) => {
                debug_assert!(range.fed);
                Ok(SplicedStreamDigest::Spine {
                    hole: self.hole,
                    first_chunk: self.first_chunk,
                    siblings: self.completed,
                    root_right: range.hasher.finalize_non_root(),
                })
            }
        }
    }
}

/// A stream's cached BLAKE3 state, replayable with new bytes in the hole.
#[derive(Clone, Debug)]
pub(crate) enum SplicedStreamDigest {
    /// The stream fits in one chunk; keep it verbatim and rehash directly.
    Short { hole: Range<usize>, stream: Vec<u8> },
    /// Chunk 0 verbatim plus the left spine's sibling chaining values,
    /// bottom-up, and the root's right-subtree chaining value.
    Spine {
        hole: Range<usize>,
        first_chunk: Vec<u8>,
        siblings: Vec<ChainingValue>,
        root_right: ChainingValue,
    },
}

impl SplicedStreamDigest {
    /// The exact `blake3::Hasher` digest of the stream with `hole_bytes`
    /// spliced into the hole.
    pub(crate) fn digest_with(&self, hole_bytes: &[u8]) -> Result<[u8; 32], SplicedDigestError> {
        let hole = match self {
            Self::Short { hole, .. } | Self::Spine { hole, .. } => hole,
        };
        if hole_bytes.len() != hole.len() {
            return Err(SplicedDigestError::HoleLengthMismatch {
                expected: hole.len(),
                actual: hole_bytes.len(),
            });
        }

        match self {
            Self::Short { hole, stream } => {
                let mut spliced = stream.clone();
                spliced[hole.clone()].copy_from_slice(hole_bytes);
                Ok(*Hasher::new().update(&spliced).finalize().as_bytes())
            }
            Self::Spine {
                hole,
                first_chunk,
                siblings,
                root_right,
            } => {
                debug_assert_eq!(first_chunk.len(), CHUNK_LEN);
                let mut spliced = first_chunk.clone();
                spliced[hole.clone()].copy_from_slice(hole_bytes);
                let mut value = Hasher::new().update(&spliced).finalize_non_root();
                for sibling in siblings {
                    value = merge_subtrees_non_root(&value, sibling, Mode::Hash);
                }
                Ok(*merge_subtrees_root(&value, root_right, Mode::Hash).as_bytes())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    use super::*;

    fn check_length(rng: &mut StdRng, stream_len: usize) {
        let stream: Vec<u8> = (0..stream_len).map(|_| rng.random()).collect();
        let hole_domain = stream_len.min(CHUNK_LEN);
        let hole_start = rng.random_range(0..hole_domain);
        let hole_end = rng.random_range(hole_start..=hole_domain);
        let hole = hole_start..hole_end;

        // Build from the placeholder stream in randomly sized pushes.
        let mut builder = SplicedDigestBuilder::new(hole.clone());
        let mut offset = 0;
        while offset < stream.len() {
            let take = rng.random_range(1..=(stream.len() - offset).min(4096));
            builder.push(&stream[offset..offset + take]);
            offset += take;
        }
        let cached = builder.finish().expect("valid hole");

        for _ in 0..2 {
            let replacement: Vec<u8> = (0..hole.len()).map(|_| rng.random()).collect();
            let mut spliced = stream.clone();
            spliced[hole.clone()].copy_from_slice(&replacement);
            let expected = *blake3::hash(&spliced).as_bytes();
            assert_eq!(
                cached.digest_with(&replacement).expect("hole length"),
                expected,
                "stream_len={stream_len} hole={hole:?}"
            );
        }
    }

    #[test]
    fn matches_plain_hasher_across_boundaries() {
        let mut rng = StdRng::seed_from_u64(0x511ced_d19357);
        let mut lengths: Vec<usize> = vec![
            1,
            2,
            41,
            42,
            58,
            CHUNK_LEN - 1,
            CHUNK_LEN,
            CHUNK_LEN + 1,
            2 * CHUNK_LEN - 1,
            2 * CHUNK_LEN,
            2 * CHUNK_LEN + 1,
            3 * CHUNK_LEN,
            3 * CHUNK_LEN + 17,
            4 * CHUNK_LEN - 1,
            4 * CHUNK_LEN,
            4 * CHUNK_LEN + 1,
            7 * CHUNK_LEN + 511,
            8 * CHUNK_LEN,
            16 * CHUNK_LEN + 7,
            64 * CHUNK_LEN,
            100_000,
            (1 << 20) - 1,
            1 << 20,
            (1 << 20) + 12_345,
        ];
        for _ in 0..8 {
            lengths.push(rng.random_range(1..300_000));
        }
        for stream_len in lengths {
            check_length(&mut rng, stream_len);
        }
    }

    #[test]
    fn rejects_hole_beyond_stream_or_chunk() {
        let mut builder = SplicedDigestBuilder::new(4..10);
        builder.push(&[0; 8]);
        assert_eq!(
            builder.finish().unwrap_err(),
            SplicedDigestError::HoleOutOfBounds {
                hole: 4..10,
                stream_len: 8,
            }
        );

        let mut builder = SplicedDigestBuilder::new(1020..1030);
        builder.push(&[0; 2048]);
        assert_eq!(
            builder.finish().unwrap_err(),
            SplicedDigestError::HoleOutOfBounds {
                hole: 1020..1030,
                stream_len: 2048,
            }
        );
    }

    #[test]
    fn rejects_wrong_replacement_length() {
        let mut builder = SplicedDigestBuilder::new(2..6);
        builder.push(&[7; 100]);
        let cached = builder.finish().expect("valid hole");
        assert_eq!(
            cached.digest_with(&[0; 3]),
            Err(SplicedDigestError::HoleLengthMismatch {
                expected: 4,
                actual: 3,
            })
        );
    }
}
