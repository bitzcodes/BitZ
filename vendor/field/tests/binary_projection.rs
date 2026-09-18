use field::{F2Poly, Gf128, Gf128Ops, PreparedGf128Projection};

fn check<const D: usize>() {
    for point in [Gf128::ZERO, Gf128::ONE, Gf128::new(0xabcdef, u64::MAX)] {
        let prepared = PreparedGf128Projection::new(point, D);
        let mask = if D == 64 { u64::MAX } else { (1u64 << D) - 1 };
        let data: Vec<_> = (0..73)
            .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .collect();
        for offset in 0..4 {
            for n in 0..=65 {
                let input = &data[offset..offset + n];
                let mut output = vec![Gf128::ZERO; n];
                prepared.project_into::<D>(input, &mut output);
                for (&bits, &actual) in input.iter().zip(&output) {
                    let polynomial = F2Poly::<64, 1>::from_polynomial_words([bits & mask]);
                    let expected = Gf128Ops.evaluate_polynomial(&polynomial, &point);
                    assert_eq!(actual, expected);
                    assert_eq!(prepared.project(bits), expected);
                }
            }
        }
    }
}

#[test]
fn prepared_projection_matches_polynomial_evaluation_for_offsets_and_tails() {
    check::<0>();
    check::<1>();
    check::<32>();
    check::<63>();
    check::<64>();
}
