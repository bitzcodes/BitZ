//! Fused sumcheck operations. Consumers retain the protocol and thread schedule.
use crate::FieldOps;

/// Messages for an inner product of two multilinear tables. A pair round
/// returns `[constant, quadratic]`; the caller derives the linear coefficient
/// from the current claim. `Folded` names the storage consumed by later rounds.
pub trait DotProductKernels<Src, Folded>: FieldOps {
    fn dot_pair_round(&self, weights: &[Self::Elem], values: &[Src]) -> [Self::Elem; 2];
    /// Reads exactly once at each ascending index in `0..weights.len()`.
    fn dot_pair_round_map(
        &self,
        weights: &[Self::Elem],
        read: impl FnMut(usize) -> Src,
    ) -> [Self::Elem; 2];
    /// Folds both inputs and computes the next round in the same traversal.
    /// Inputs have twice the output length; the output length must be even.
    fn dot_fold_round_into(
        &self,
        weights: &[Self::Elem],
        values: &[Src],
        weights_out: &mut [Self::Elem],
        values_out: &mut [Folded],
        challenge: &Self::Elem,
    ) -> [Self::Elem; 2];
    fn dot_fold_round_map_into(
        &self,
        weights: &[Self::Elem],
        read: impl FnMut(usize) -> Src,
        weights_out: &mut [Self::Elem],
        values_out: &mut [Folded],
        challenge: &Self::Elem,
    ) -> [Self::Elem; 2];
}

/// Weighted quadratic messages over adjacent pairs. Coefficients are returned
/// in monomial order `[constant, linear, quadratic]`. All shape checks are
/// public; in-place methods write only the documented prefix.
pub trait SumcheckKernels: FieldOps {
    fn eqf_single_pair_round(
        &self,
        l: &[Self::Elem],
        r: &[Self::Elem],
        weights: &[Self::Elem],
        half: usize,
    ) -> [Self::Elem; 3];
    fn eqf_two_pair_round(
        &self,
        l0: &[Self::Elem],
        r0: &[Self::Elem],
        l1: &[Self::Elem],
        r1: &[Self::Elem],
        weights: &[Self::Elem],
        half: usize,
    ) -> [Self::Elem; 3];
    fn eqf_fold_in_place(&self, values: &mut [Self::Elem], challenge: &Self::Elem, half: usize);
    /// Fold 4*half values into the 2*half prefix and compute the next message
    /// in the same traversal. Values above that prefix are left unchanged.
    fn eqf_fused_fold_round(
        &self,
        l: &mut [Self::Elem],
        r: &mut [Self::Elem],
        challenge: &Self::Elem,
        weights: &[Self::Elem],
        half: usize,
    ) -> [Self::Elem; 3];
    /// Consume at most two deferred challenges, retain four folded values per
    /// quad, and return the existing X1-monomial / X2-node layout at index u*3+v.
    fn eqf_grid_pass(
        &self,
        l: &mut [Self::Elem],
        r: &mut [Self::Elem],
        pending: &[Self::Elem],
        suffix: &[Self::Elem],
        quads: usize,
    ) -> [Self::Elem; 9];
}

pub(crate) trait RoundArithmetic: FieldOps + PreparedRoundMul {
    type Acc;
    fn zero_acc(&self) -> Self::Acc;
    fn mac(&self, acc: &mut Self::Acc, a: &Self::Elem, b: &Self::Elem);
    fn finish(&self, acc: Self::Acc) -> Self::Elem;
}

/// Fixed multipliers are prepared once per pass, outside its element loop.
pub(crate) trait PreparedRoundMul: FieldOps {
    type Multiplier;
    fn prepare_round_mul(&self, value: &Self::Elem) -> Self::Multiplier;
    fn mul_round_prepared(&self, multiplier: &Self::Multiplier, value: &Self::Elem) -> Self::Elem;
}

fn required(count: usize, arity: usize) -> usize {
    count.checked_mul(arity).expect("sumcheck shape overflow")
}
fn check_pair<C: FieldOps + ?Sized>(
    l: &[C::Elem],
    r: &[C::Elem],
    weights: &[C::Elem],
    half: usize,
    arity: usize,
) {
    let n = required(half, arity);
    assert!(l.len() >= n && r.len() >= n, "insufficient sumcheck input");
    assert!(weights.len() >= half, "insufficient sumcheck weights");
}
fn fold<C: PreparedRoundMul>(
    field: &C,
    a: C::Elem,
    b: C::Elem,
    challenge: &C::Multiplier,
) -> C::Elem {
    field.add(&a, &field.mul_round_prepared(challenge, &field.sub(&b, &a)))
}
fn slot<C: RoundArithmetic>(
    field: &C,
    acc: &mut [C::Acc; 3],
    l: [C::Elem; 2],
    r: [C::Elem; 2],
    weight: &C::Elem,
) {
    // These two intermediate products must be reduced before multiplying
    // again. Only products feeding the coefficient sums remain unreduced.
    let left = [field.mul(weight, &l[0]), field.mul(weight, &l[1])];
    field.mac(&mut acc[0], &left[0], &r[0]);
    field.mac(&mut acc[1], &left[1], &r[1]);
    field.mac(
        &mut acc[2],
        &field.sub(&left[1], &left[0]),
        &field.sub(&r[1], &r[0]),
    );
}
fn finish<C: RoundArithmetic>(field: &C, acc: [C::Acc; 3]) -> [C::Elem; 3] {
    let [a, b, c] = acc.map(|a| field.finish(a));
    [a, field.sub(&field.sub(&b, &a), &c), c]
}
pub(crate) fn single<C: RoundArithmetic>(
    field: &C,
    l: &[C::Elem],
    r: &[C::Elem],
    weights: &[C::Elem],
    half: usize,
) -> [C::Elem; 3] {
    check_pair::<C>(l, r, weights, half, 2);
    let mut acc = core::array::from_fn(|_| field.zero_acc());
    for ((l, r), w) in l[..2 * half]
        .chunks_exact(2)
        .zip(r.chunks_exact(2))
        .zip(weights)
    {
        slot(field, &mut acc, [l[0], l[1]], [r[0], r[1]], w);
    }
    finish(field, acc)
}
pub(crate) fn two<C: RoundArithmetic>(
    field: &C,
    l0: &[C::Elem],
    r0: &[C::Elem],
    l1: &[C::Elem],
    r1: &[C::Elem],
    weights: &[C::Elem],
    half: usize,
) -> [C::Elem; 3] {
    check_pair::<C>(l0, r0, weights, half, 2);
    check_pair::<C>(l1, r1, weights, half, 2);
    let mut acc = core::array::from_fn(|_| field.zero_acc());
    for i in 0..half {
        slot(
            field,
            &mut acc,
            [l0[2 * i], l0[2 * i + 1]],
            [r0[2 * i], r0[2 * i + 1]],
            &weights[i],
        );
        slot(
            field,
            &mut acc,
            [l1[2 * i], l1[2 * i + 1]],
            [r1[2 * i], r1[2 * i + 1]],
            &weights[i],
        );
    }
    finish(field, acc)
}
pub(crate) fn fold_in_place<C: PreparedRoundMul>(
    field: &C,
    values: &mut [C::Elem],
    challenge: &C::Elem,
    half: usize,
) {
    assert!(required(half, 2) <= values.len(), "insufficient fold input");
    let challenge = field.prepare_round_mul(challenge);
    for i in 0..half {
        values[i] = fold(field, values[2 * i], values[2 * i + 1], &challenge);
    }
}
pub(crate) fn fused<C: RoundArithmetic>(
    field: &C,
    l: &mut [C::Elem],
    r: &mut [C::Elem],
    challenge: &C::Elem,
    weights: &[C::Elem],
    half: usize,
) -> [C::Elem; 3] {
    check_pair::<C>(l, r, weights, half, 4);
    let challenge = field.prepare_round_mul(challenge);
    let mut acc = core::array::from_fn(|_| field.zero_acc());
    for i in 0..half {
        let left = [
            fold(field, l[4 * i], l[4 * i + 1], &challenge),
            fold(field, l[4 * i + 2], l[4 * i + 3], &challenge),
        ];
        let right = [
            fold(field, r[4 * i], r[4 * i + 1], &challenge),
            fold(field, r[4 * i + 2], r[4 * i + 3], &challenge),
        ];
        l[2 * i..2 * i + 2].copy_from_slice(&left);
        r[2 * i..2 * i + 2].copy_from_slice(&right);
        slot(field, &mut acc, left, right, &weights[i]);
    }
    finish(field, acc)
}
fn logical<C: PreparedRoundMul>(
    field: &C,
    values: &[C::Elem],
    index: usize,
    pending: &[Option<C::Multiplier>],
) -> C::Elem {
    let start = index << pending.len();
    match pending {
        [] => values[start],
        [Some(r)] => fold(field, values[start], values[start + 1], r),
        [Some(r), Some(s)] => fold(
            field,
            fold(field, values[start], values[start + 1], r),
            fold(field, values[start + 2], values[start + 3], r),
            s,
        ),
        _ => unreachable!("public pending count was checked"),
    }
}
fn grid<C: FieldOps>(field: &C, a: [C::Elem; 4]) -> [C::Elem; 9] {
    let d0 = field.sub(&a[1], &a[0]);
    let d1 = field.sub(&a[3], &a[2]);
    [
        a[0],
        a[1],
        d0,
        a[2],
        a[3],
        d1,
        field.sub(&a[2], &a[0]),
        field.sub(&a[3], &a[1]),
        field.sub(&d1, &d0),
    ]
}
pub(crate) fn grid_pass<C: RoundArithmetic>(
    field: &C,
    l: &mut [C::Elem],
    r: &mut [C::Elem],
    pending: &[C::Elem],
    suffix: &[C::Elem],
    quads: usize,
) -> [C::Elem; 9] {
    assert!(pending.len() <= 2, "at most two deferred folds");
    check_pair::<C>(l, r, suffix, quads, 4 << pending.len());
    // Two stack slots suffice for the public maximum; only actual pending
    // multipliers are prepared. No allocation or per-element setup is hidden.
    let prepared: [Option<C::Multiplier>; 2] =
        core::array::from_fn(|i| pending.get(i).map(|v| field.prepare_round_mul(v)));
    let multipliers = &prepared[..pending.len()];
    let mut acc: [C::Acc; 9] = core::array::from_fn(|_| field.zero_acc());
    for b in 0..quads {
        let left = core::array::from_fn(|i| logical(field, l, 4 * b + i, multipliers));
        let right = core::array::from_fn(|i| logical(field, r, 4 * b + i, multipliers));
        if !pending.is_empty() {
            l[4 * b..4 * b + 4].copy_from_slice(&left);
            r[4 * b..4 * b + 4].copy_from_slice(&right);
        }
        let left = grid(field, left.map(|v| field.mul(&suffix[b], &v)));
        let right = grid(field, right);
        for i in 0..9 {
            field.mac(&mut acc[i], &left[i], &right[i]);
        }
    }
    let nodes = acc.map(|a| field.finish(a));
    core::array::from_fn(|i| {
        let (u, v) = (i / 3, i % 3);
        let b = 3 * v;
        match u {
            0 => nodes[b],
            2 => nodes[b + 2],
            _ => field.sub(&field.sub(&nodes[b + 1], &nodes[b]), &nodes[b + 2]),
        }
    })
}

macro_rules! implement_sumcheck {
    ([$($generic:tt)*] $provider:ty) => {
        impl<$($generic)*> $crate::batch::SumcheckKernels for $provider {
            fn eqf_single_pair_round(&self,l:&[Self::Elem],r:&[Self::Elem],w:&[Self::Elem],n:usize)->[Self::Elem;3] {$crate::batch::single(self,l,r,w,n)}
            fn eqf_two_pair_round(&self,l0:&[Self::Elem],r0:&[Self::Elem],l1:&[Self::Elem],r1:&[Self::Elem],w:&[Self::Elem],n:usize)->[Self::Elem;3] {$crate::batch::two(self,l0,r0,l1,r1,w,n)}
            fn eqf_fold_in_place(&self,v:&mut[Self::Elem],r:&Self::Elem,n:usize) {$crate::batch::fold_in_place(self,v,r,n)}
            fn eqf_fused_fold_round(&self,l:&mut[Self::Elem],r:&mut[Self::Elem],rho:&Self::Elem,w:&[Self::Elem],n:usize)->[Self::Elem;3] {$crate::batch::fused(self,l,r,rho,w,n)}
            fn eqf_grid_pass(&self,l:&mut[Self::Elem],r:&mut[Self::Elem],p:&[Self::Elem],s:&[Self::Elem],n:usize)->[Self::Elem;9] {$crate::batch::grid_pass(self,l,r,p,s,n)}
        }
    };
}
pub(crate) use implement_sumcheck;
