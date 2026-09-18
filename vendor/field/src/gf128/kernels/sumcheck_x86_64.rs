//! Fused deferred folds and weighted grid accumulation using the existing
//! PCLMUL field arithmetic. Traversal and parallelism belong to the caller.
use crate::Gf128;
use crate::gf128::x86_64::{wide_add, wide_mul, wide_reduce, wide_zero};

#[inline(always)]
fn fold<const D: usize>(v: &[Gf128], i: usize, challenges: &[Gf128; 3]) -> Gf128 {
    let base = i << D;
    let a = v[base];
    if D == 0 {
        a
    } else if D == 1 {
        a + Gf128::from(wide_reduce(wide_mul(
            *challenges[0].as_words(),
            *(a + v[base + 1]).as_words(),
        )))
    } else {
        let b = v[base + 1];
        let c = v[base + 2];
        let d = v[base + 3];
        // a + r0(a+b) + r1(a+c) + r0*r1(a+b+c+d). Reduction is
        // linear, so the three products need only one final reduction.
        let p0 = wide_mul(*challenges[0].as_words(), *(a + b).as_words());
        let p1 = wide_mul(*challenges[1].as_words(), *(a + c).as_words());
        let p01 = wide_mul(*challenges[2].as_words(), *(a + b + c + d).as_words());
        a + Gf128::from(wide_reduce(wide_add(wide_add(p0, p1), p01)))
    }
}

#[inline(always)]
fn nodes(a: [Gf128; 4]) -> [Gf128; 9] {
    let d0 = a[1] + a[0];
    let d1 = a[3] + a[2];
    [
        a[0],
        a[1],
        d0,
        a[2],
        a[3],
        d1,
        a[2] + a[0],
        a[3] + a[1],
        d1 + d0,
    ]
}

fn pass<const D: usize>(
    l: &mut [Gf128],
    r: &mut [Gf128],
    challenges: [Gf128; 3],
    suffix: &[Gf128],
    quads: usize,
) -> [Gf128; 9] {
    let mut acc = [wide_zero(); 9];
    for (b, weight) in suffix[..quads].iter().enumerate() {
        // Finish loading both physical quads before landing their logical
        // values in the prefix; stores must never overtake unread input.
        let lv = core::array::from_fn(|i| fold::<D>(l, 4 * b + i, &challenges));
        let rv = core::array::from_fn(|i| fold::<D>(r, 4 * b + i, &challenges));
        if D != 0 {
            l[4 * b..4 * b + 4].copy_from_slice(&lv);
            r[4 * b..4 * b + 4].copy_from_slice(&rv);
        }
        let left = nodes(lv.map(|v| *weight * v));
        let right = nodes(rv);
        for i in 0..9 {
            acc[i] = wide_add(acc[i], wide_mul(*left[i].as_words(), *right[i].as_words()));
        }
    }
    let values = acc.map(|a| Gf128::from(wide_reduce(a)));
    core::array::from_fn(|i| {
        let (u, v) = (i / 3, i % 3);
        let base = 3 * v;
        match u {
            0 => values[base],
            2 => values[base + 2],
            _ => values[base + 1] + values[base] + values[base + 2],
        }
    })
}

pub(super) fn grid_pass(
    l: &mut [Gf128],
    r: &mut [Gf128],
    pending: &[Gf128],
    suffix: &[Gf128],
    quads: usize,
) -> [Gf128; 9] {
    match pending {
        [] => pass::<0>(l, r, [Gf128::ZERO; 3], suffix, quads),
        &[r0] => pass::<1>(l, r, [r0, Gf128::ZERO, Gf128::ZERO], suffix, quads),
        &[r0, r1] => pass::<2>(l, r, [r0, r1, r0 * r1], suffix, quads),
        _ => unreachable!("grid pass supports at most two deferred folds"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grid_matches_generic_including_untouched_storage() {
        let mut state = 0x73a27ac43bu64;
        let mut sample = || {
            let mut word = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            Gf128::from([word(), word()])
        };
        for quads in [0, 1, 2, 3, 17, 64, 1025] {
            for count in 0..=2 {
                for special in [
                    None,
                    Some(Gf128::ZERO),
                    Some(Gf128::ONE),
                    Some(Gf128::from([u64::MAX; 2])),
                ] {
                    let pending: Vec<_> = (0..count)
                        .map(|_| special.unwrap_or_else(&mut sample))
                        .collect();
                    let len = (4 * quads << count) + 7;
                    let mut l: Vec<_> = (0..len).map(|_| sample()).collect();
                    let mut r: Vec<_> = (0..len).map(|_| sample()).collect();
                    let suffix: Vec<_> = (0..quads).map(|_| sample()).collect();
                    let (mut expected_l, mut expected_r) = (l.clone(), r.clone());
                    let expected = crate::batch::grid_pass(
                        &crate::Gf128Ops,
                        &mut expected_l,
                        &mut expected_r,
                        &pending,
                        &suffix,
                        quads,
                    );
                    assert_eq!(
                        grid_pass(&mut l, &mut r, &pending, &suffix, quads),
                        expected
                    );
                    assert_eq!(l, expected_l);
                    assert_eq!(r, expected_r);
                }
            }
        }
    }
}
