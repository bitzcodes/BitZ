//! Current production entrypoints and the public generic API, without arithmetic copies.
use super::*;
use crate::sumcheck::{
    UngrindedRoundBoundary,
    outer,
};
pub(super) const REVISION: &str = "current";
pub(super) fn prepare(f: &Field, k: usize) -> Option<outer::PreparedUnivariateSkip<Elem>> {
    (k > 0).then(|| outer::prepare_univariate_skip(f, k as u8).unwrap())
}
pub(super) fn production(
    input: &Inputs,
    f: &Field,
    t: &mut Blake3Transcript,
    tau: &[Elem],
    n: usize,
    k: usize,
) -> Proof {
    let (lo, hi) = super::super::raw_monty::make_equality_factors_raw(f, tau);
    macro_rules! prove {
        ($rows:expr) => {
            if k == 0 {
                Proof::Ordinary(
                    crate::sumcheck::outer::prove_outer_sumcheck(
f,
t,
crate::sumcheck::outer::OuterClaim::RowwiseZero,
tau,
$rows,
Some(crate::sumcheck::outer::arithmetic::factors_from_raw(f, lo, hi)),
&mut crate::sumcheck::UngrindedRoundBoundary,
).map(crate::sumcheck::proof::OuterSumcheckOutput::from)
                        .unwrap()
                        .proof,
                )
            } else {
                Proof::Skip(
                    crate::sumcheck::outer::prepare_univariate_skip(f, k as u8).and_then(|prepared| crate::sumcheck::outer::prove_outer_zerocheck_with_skip(
f,
t,
&prepared,
tau,
$rows,
Some(crate::sumcheck::outer::arithmetic::factors_from_raw(f, lo, hi)),
&mut crate::sumcheck::UngrindedRoundBoundary,
)).map(crate::sumcheck::outer::univariate::UnivariateSkipOuterSumcheckOutput::from)
                        .unwrap()
                        .proof,
                )
            }
        };
    }
    match input {
        Inputs::U32 { a, b, c, .. } => prove!(NativeProducts {
            az: a,
            bz: b,
            cz: c
        }),
        Inputs::U64 {
            a,
            b,
            lo: cl,
            hi: ch,
            ..
        } => prove!(NativeWideProducts::new(a, b, cl, ch, 1 << n)),
        Inputs::U128 {
            a,
            b,
            lo: cl,
            hi: ch,
            ..
        } => prove!(NativeWideProducts::new(a, b, cl, ch, 1 << n)),
    }
}

pub(super) fn generic(
    input: &Inputs,
    f: &Field,
    t: &mut Blake3Transcript,
    tau: &[Elem],
    k: usize,
    prepared: &Option<outer::PreparedUnivariateSkip<Elem>>,
) -> Proof {
    let wrap = |o: outer::OuterOutput<Elem>| OuterSumcheckProof {
        sumcheck: o.proof,
        az_mle_claim: o.evaluations.ax,
        bz_mle_claim: o.evaluations.bx,
        cz_mle_claim: o.evaluations.cx,
    };
    macro_rules! prove {
        ($a:expr,$b:expr,$c:expr) => {
            if k == 0 {
                Proof::Ordinary(wrap(
                    crate::sumcheck::outer::prove_outer_sumcheck(
                        f,
                        t,
                        crate::sumcheck::outer::OuterClaim::RowwiseZero,
                        tau,
                        crate::sumcheck::outer::OuterSlices {
                            ax: $a,
                            bx: $b,
                            cx: $c,
                        },
                        None,
                        &mut UngrindedRoundBoundary,
                    )
                    .unwrap(),
                ))
            } else {
                let o = crate::sumcheck::outer::prove_outer_zerocheck_with_skip(
                    f,
                    t,
                    prepared.as_ref().unwrap(),
                    tau,
                    crate::sumcheck::outer::OuterSlices {
                        ax: $a,
                        bx: $b,
                        cx: $c,
                    },
                    None,
                    &mut UngrindedRoundBoundary,
                )
                .unwrap();
                Proof::Skip(UnivariateSkipOuterSumcheckProof {
                    skip: o.prefix,
                    tail: wrap(o.tail),
                })
            }
        };
    }
    match input {
        Inputs::U32 {
            narrow_a,
            narrow_b,
            c,
            ..
        } => prove!(narrow_a, narrow_b, c),
        Inputs::U64 { a, b, c, .. } => prove!(a, b, c),
        Inputs::U128 { a, b, c, .. } => prove!(a, b, c),
    }
}

pub(super) fn reset_measurements() {
    crate::sumcheck::outer::measure::reset();
}
pub(super) fn take_measurements() -> Option<[u64; 4]> {
    crate::sumcheck::outer::measure::take()
}
