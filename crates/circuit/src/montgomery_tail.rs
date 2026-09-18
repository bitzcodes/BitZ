//! Owned Montgomery coefficients with geometric runs and implicit zero gaps.
use crate::linear_map::circuit::{add_representatives, mul_representatives};
use core::ops::Range;
use field::{FpCtx, Uint};

#[derive(Clone, Debug)]
pub enum TailSegment {
    Geometric {
        start: usize,
        len: usize,
        base: [u64; 2],
    },
    Literals {
        start: usize,
        values: Range<usize>,
    },
}
impl TailSegment {
    pub fn start(&self) -> usize {
        match self {
            Self::Geometric { start, .. } | Self::Literals { start, .. } => *start,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Geometric { len, .. } => *len,
            Self::Literals { values, .. } => values.len(),
        }
    }
}

/// Segments are authoritative. Indices absent from them have coefficient zero.
/// All words are canonical representatives of Montgomery residues for `field`.
pub struct MontgomeryTail {
    field: FpCtx<2>,
    len: usize,
    segments: Vec<TailSegment>,
    literals: Vec<[u64; 2]>,
}
impl MontgomeryTail {
    pub fn new(
        field: FpCtx<2>,
        len: usize,
        segments: Vec<TailSegment>,
        literals: Vec<[u64; 2]>,
    ) -> Result<Self, &'static str> {
        let valid = |v: &[u64; 2]| u128::from(Uint::from_words(*v)) < u128::from(*field.modulus());
        if !literals.iter().all(valid) {
            return Err("noncanonical Montgomery literal");
        }
        let mut end = 0;
        let mut literal_end = 0;
        for segment in &segments {
            let size = segment.len();
            if size == 0 || segment.start() < end {
                return Err("empty or overlapping tail segment");
            }
            end = segment
                .start()
                .checked_add(size)
                .filter(|&end| end <= len)
                .ok_or("tail segment out of bounds")?;
            match segment {
                TailSegment::Geometric { base, .. } => {
                    if !valid(base) {
                        return Err("noncanonical Montgomery base");
                    }
                }
                TailSegment::Literals { values, .. } => {
                    if values.start != literal_end || values.end > literals.len() {
                        return Err("invalid literal coverage");
                    }
                    literal_end = values.end;
                }
            }
        }
        if literal_end != literals.len() {
            return Err("unused tail literals");
        }
        Ok(Self {
            field,
            len,
            segments,
            literals,
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn field(&self) -> &FpCtx<2> {
        &self.field
    }
    pub fn segments(&self) -> &[TailSegment] {
        &self.segments
    }
    pub fn literals(&self) -> &[[u64; 2]] {
        &self.literals
    }

    fn advance(&self, mut base: [u64; 2], mut power: usize) -> [u64; 2] {
        // Shift by a public exponent in logarithmic time. Doubling preserves
        // Montgomery form, so no division or integer/field conversion occurs.
        if power == 0 {
            return base;
        }
        let one = *self
            .field
            .from_canonical_integer(&Uint::from(1u64))
            .as_montgomery_integer()
            .as_words();
        let mut factor = add_representatives(one, one, &self.field);
        while power != 0 {
            if power & 1 != 0 {
                base = mul_representatives(base, factor, &self.field);
            }
            power >>= 1;
            if power != 0 {
                factor = mul_representatives(factor, factor, &self.field);
            }
        }
        base
    }
    pub fn value(&self, index: usize) -> Option<[u64; 2]> {
        if index >= self.len {
            return None;
        }
        let i = self.segments.partition_point(|s| s.start() <= index);
        let Some(s) = i.checked_sub(1).map(|i| &self.segments[i]) else {
            return Some([0; 2]);
        };
        if index >= s.start() + s.len() {
            return Some([0; 2]);
        }
        Some(match s {
            TailSegment::Geometric { start, base, .. } => self.advance(*base, index - start),
            TailSegment::Literals { start, values } => self.literals[values.start + index - start],
        })
    }

    /// Visit nonzero-structure intersections with a range, in index order.
    /// Literals are length-one runs. No expanded coefficient array is built.
    pub fn visit(&self, range: Range<usize>, mut emit: impl FnMut(usize, usize, [u64; 2])) {
        assert!(range.start <= range.end && range.end <= self.len);
        if range.is_empty() {
            return;
        }
        let first = self
            .segments
            .partition_point(|s| s.start() + s.len() <= range.start);
        for s in &self.segments[first..] {
            if s.start() >= range.end {
                break;
            }
            let lo = s.start().max(range.start);
            let hi = (s.start() + s.len()).min(range.end);
            match s {
                TailSegment::Geometric { start, base, .. } => {
                    emit(lo, hi - lo, self.advance(*base, lo - start))
                }
                TailSegment::Literals { start, values } => {
                    for i in lo..hi {
                        emit(i, 1, self.literals[values.start + i - start]);
                    }
                }
            }
        }
    }

    /// Add public coefficients. Geometric runs are split at exceptions; the
    /// original dense table is never materialized. Duplicate indices add.
    pub fn add_sparse(&mut self, updates: &[(usize, [u64; 2])]) -> Result<(), &'static str> {
        if updates.is_empty() {
            return Ok(());
        }
        if updates.iter().any(|(i, v)| {
            *i >= self.len || u128::from(Uint::from_words(*v)) >= u128::from(*self.field.modulus())
        }) {
            return Err("invalid public coefficient adjustment");
        }
        let mut updates = updates.to_vec();
        updates.sort_by_key(|x| x.0);
        let mut segments = Vec::with_capacity(self.segments.len() + 2 * updates.len());
        let mut literals = Vec::with_capacity(self.literals.len() + updates.len());
        let mut cursor = 0;
        let mut u = 0;
        let append = |start: usize,
                      len: usize,
                      base: [u64; 2],
                      segments: &mut Vec<TailSegment>,
                      literals: &mut Vec<[u64; 2]>| {
            if len == 1 {
                let at = literals.len();
                literals.push(base);
                if let Some(TailSegment::Literals {
                    start: prev,
                    values,
                }) = segments.last_mut()
                {
                    if *prev + values.len() == start && values.end == at {
                        values.end += 1;
                        return;
                    }
                }
                segments.push(TailSegment::Literals {
                    start,
                    values: at..at + 1,
                });
            } else {
                segments.push(TailSegment::Geometric { start, len, base });
            }
        };
        while u < updates.len() {
            let index = updates[u].0;
            self.visit(cursor..index, |s, n, b| {
                append(s, n, b, &mut segments, &mut literals)
            });
            let mut value = self.value(index).expect("validated index");
            while u < updates.len() && updates[u].0 == index {
                value = add_representatives(value, updates[u].1, &self.field);
                u += 1;
            }
            append(index, 1, value, &mut segments, &mut literals);
            cursor = index + 1;
        }
        self.visit(cursor..self.len, |s, n, b| {
            append(s, n, b, &mut segments, &mut literals)
        });
        *self = Self::new(self.field.clone(), self.len, segments, literals)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validated_segments_and_public_adjustments_match_dense_integers() {
        for prime in [17u128, 97, (1u128 << 127) - 1] {
            let field = field::create_prime_field(Uint::from(prime));
            let encode = |n: u128| {
                *field
                    .from_canonical_integer(&Uint::from(n % prime))
                    .as_montgomery_integer()
                    .as_words()
            };
            let decode = |words: [u64; 2]| {
                let n = field.to_integer(&field.from_montgomery_integer(Uint::from_words(words)));
                u128::from(n.as_words()[0]) | (u128::from(n.as_words()[1]) << 64)
            };
            let mut dense = vec![0u128; 85];
            let mut value = 3;
            for v in &mut dense[3..71] {
                *v = value;
                value = (value * 2) % prime;
            }
            dense[73] = 5;
            dense[74] = 7;
            let mut tail = MontgomeryTail::new(
                field.clone(),
                85,
                vec![
                    TailSegment::Geometric {
                        start: 3,
                        len: 68,
                        base: encode(3),
                    },
                    TailSegment::Literals {
                        start: 73,
                        values: 0..2,
                    },
                ],
                vec![encode(5), encode(7)],
            )
            .unwrap();
            for (i, &v) in dense.iter().enumerate() {
                assert_eq!(decode(tail.value(i).unwrap()), v);
            }
            assert!(tail.value(85).is_none());
            let updates = [
                (3, 11),
                (40, 13),
                (40, 1),
                (41, 2),
                (70, 2),
                (74, 9),
                (0, 6),
                (84, 8),
            ];
            for &(i, n) in &updates {
                dense[i] = (dense[i] + n) % prime;
            }
            tail.add_sparse(&updates.map(|(i, n)| (i, encode(n))))
                .unwrap();
            for width in [1usize, 2, 4, 8, 16, 31, 100] {
                let mut restored = vec![0u128; 85];
                for start in (0..85).step_by(width) {
                    tail.visit(start..85.min(start + width), |at, len, base| {
                        let mut v = decode(base);
                        for out in &mut restored[at..at + len] {
                            *out = v;
                            v = (v * 2) % prime;
                        }
                    });
                }
                assert_eq!(restored, dense);
            }
            assert!(
                MontgomeryTail::new(
                    field.clone(),
                    4,
                    vec![TailSegment::Geometric {
                        start: 3,
                        len: 2,
                        base: encode(1)
                    }],
                    vec![]
                )
                .is_err()
            );
            assert!(
                MontgomeryTail::new(
                    field.clone(),
                    4,
                    vec![TailSegment::Geometric {
                        start: 0,
                        len: 0,
                        base: encode(1)
                    }],
                    vec![]
                )
                .is_err()
            );
            assert!(
                MontgomeryTail::new(
                    field.clone(),
                    4,
                    vec![TailSegment::Geometric {
                        start: 0,
                        len: 2,
                        base: *Uint::from(prime).as_words()
                    }],
                    vec![]
                )
                .is_err()
            );
            assert!(
                MontgomeryTail::new(
                    field.clone(),
                    4,
                    vec![TailSegment::Literals {
                        start: 0,
                        values: 1..2
                    }],
                    vec![encode(1), encode(1)]
                )
                .is_err()
            );
            assert!(
                MontgomeryTail::new(
                    field.clone(),
                    4,
                    vec![
                        TailSegment::Geometric {
                            start: 0,
                            len: 3,
                            base: encode(1)
                        },
                        TailSegment::Geometric {
                            start: 2,
                            len: 2,
                            base: encode(1)
                        }
                    ],
                    vec![]
                )
                .is_err()
            );
            assert!(tail.add_sparse(&[(85, encode(1))]).is_err());
        }
    }
}
