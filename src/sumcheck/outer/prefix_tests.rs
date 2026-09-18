//! Independent integer interpolation oracle for every supported skip width.
use super::{OuterArithmetic, univariate_api::exterior_values};
use field::{FpCtx, IntegerEmbedding, Uint, Z};
use num_bigint::BigInt;
use num_traits::{One, Zero};

fn signed<const N: usize>(v: Z<N>) -> BigInt {
    BigInt::from_signed_bytes_le(
        &v.as_words()
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn interpolate(values: &[BigInt], node: i64) -> BigInt {
    values
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let mut numerator = BigInt::one();
            let mut denominator = BigInt::one();
            for j in 0..values.len() {
                if i != j {
                    numerator *= node - j as i64;
                    denominator *= i as i64 - j as i64;
                }
            }
            assert!((&numerator % &denominator).is_zero());
            value * (numerator / denominator)
        })
        .sum()
}
fn check<A: Copy, C: Copy, const M: usize, const LANES: usize, const STEPS: usize>(
    av: [A; M],
    bv: [A; M],
    cv: [C; M],
    big_ab: impl Fn(<FpCtx<2> as OuterArithmetic<A, C>>::WidenedAB) -> BigInt,
    big_c: impl Fn(<FpCtx<2> as OuterArithmetic<A, C>>::WidenedC) -> BigInt,
    big_r: impl Fn(<FpCtx<2> as OuterArithmetic<A, C>>::Residual) -> BigInt,
) where
    FpCtx<2>: OuterArithmetic<A, C>,
{
    let modulus = (1u128 << 100) - 15;
    let f = field::create_prime_field(Uint::from(modulus));
    macro_rules! op { ($method:ident $(, $x:expr)*) => { <FpCtx<2> as OuterArithmetic<A,C>>::$method(&f $(, $x)*) }; }
    let aa = av.map(|v| big_ab(op!(lift_ab, v)));
    let bb = bv.map(|v| big_ab(op!(lift_ab, v)));
    let cc = cv.map(|v| big_c(op!(lift_c, v)));
    let a = exterior_values::<_, M, LANES, STEPS>(
        av.map(|v| op!(lift_ab, v)),
        |x, y| op!(add_ab, x, y),
        |x, y| op!(sub_ab, x, y),
    );
    let b = exterior_values::<_, M, LANES, STEPS>(
        bv.map(|v| op!(lift_ab, v)),
        |x, y| op!(add_ab, x, y),
        |x, y| op!(sub_ab, x, y),
    );
    let c = exterior_values::<_, M, LANES, STEPS>(
        cv.map(|v| op!(lift_c, v)),
        |x, y| op!(add_c, x, y),
        |x, y| op!(sub_c, x, y),
    );
    // A degree-M-1 polynomial's leading forward difference has binomial weights.
    let leading = |values: &[BigInt]| {
        let mut choose = 1u64;
        let mut sum = BigInt::zero();
        for (i, v) in values.iter().enumerate() {
            sum += v * BigInt::from(choose) * if (M - 1 - i) % 2 == 0 { 1 } else { -1 };
            if i + 1 < M {
                choose = choose * (M - 1 - i) as u64 / (i + 1) as u64;
            }
        }
        sum
    };
    assert_eq!(big_ab(a[LANES - 1]), leading(&aa));
    assert_eq!(big_ab(b[LANES - 1]), leading(&bb));
    assert_eq!(
        big_r(op!(product, a[LANES - 1], b[LANES - 1])),
        leading(&aa) * leading(&bb)
    );
    let w = f.from_integer(&u128::MAX);
    let weights = op!(prepare_weights, &[w]);
    for step in 1..M / 2 {
        for (node, ax, bx, cx) in [
            (
                -(step as i64),
                a[2 * (step - 1)],
                b[2 * (step - 1)],
                c[2 * (step - 1)],
            ),
            (
                (M - 1 + step) as i64,
                a[2 * (step - 1) + 1],
                b[2 * (step - 1) + 1],
                c[2 * (step - 1) + 1],
            ),
        ] {
            let ea = interpolate(&aa, node);
            let eb = interpolate(&bb, node);
            let ec = interpolate(&cc, node);
            assert_eq!(big_ab(ax), ea);
            assert_eq!(big_ab(bx), eb);
            assert_eq!(big_c(cx), ec);
            let product = op!(product, ax, bx);
            assert_eq!(big_r(product), &ea * &eb);
            let residual = op!(residual, product, cx);
            let expected = ea * eb - ec;
            assert_eq!(big_r(residual), expected);
            let mut acc = op!(accumulator);
            <FpCtx<2> as OuterArithmetic<A, C>>::accumulate::<true>(
                &f, &mut acc, &weights, 0, residual,
            );
            let actual = op!(finish_accumulator, &weights, acc);
            let q = BigInt::from(modulus);
            let expected = expected * BigInt::from(u128::from(f.to_integer(&w)));
            assert_eq!(
                BigInt::from(u128::from(f.to_integer(&actual))),
                (expected % &q + &q) % &q
            );
        }
    }
}
fn width<const M: usize, const LANES: usize, const STEPS: usize>() {
    let wide: [Uint<64>; M] = core::array::from_fn(|i| {
        Uint::from_words(core::array::from_fn(|j| {
            [u64::MAX, 0, 1, 1 << 63][(i + j) % 4]
        }))
    });
    check::<_, _, M, LANES, STEPS>(
        wide,
        core::array::from_fn(|i| wide[(i * 3 + 1) % M]),
        wide,
        signed,
        signed,
        signed,
    );
    let a32 = core::array::from_fn(|i| {
        [
            u32::MAX,
            0,
            1,
            1 << 31,
            u32::MAX - 1,
            17,
            1 << 16,
            (1 << 16) - 1,
        ][i % 8]
    });
    let b32 = core::array::from_fn(|i| a32[(i * 3 + 1) % M]);
    check::<_, _, M, LANES, STEPS>(
        a32,
        b32,
        core::array::from_fn(|i| u64::MAX - i as u64),
        BigInt::from,
        BigInt::from,
        BigInt::from,
    );
    let a64 = core::array::from_fn(|i| {
        [
            u64::MAX,
            0,
            1,
            1 << 63,
            u64::MAX - 1,
            17,
            1 << 32,
            (1 << 32) - 1,
        ][i % 8]
    });
    let b64 = core::array::from_fn(|i| a64[(i * 3 + 1) % M]);
    check::<_, _, M, LANES, STEPS>(
        a64,
        b64,
        core::array::from_fn(|i| u128::MAX - i as u128),
        BigInt::from,
        signed,
        signed,
    );
    let a128 = core::array::from_fn(|i| {
        [
            u128::MAX,
            0,
            1,
            1 << 127,
            u128::MAX - 1,
            17,
            1 << 64,
            (1 << 64) - 1,
        ][i % 8]
    });
    let b128 = core::array::from_fn(|i| a128[(i * 3 + 1) % M]);
    let c256 = core::array::from_fn(|i| {
        Uint::from_words([u64::MAX - i as u64, u64::MAX, u64::MAX, u64::MAX])
    });
    check::<_, _, M, LANES, STEPS>(a128, b128, c256, signed, signed, signed);
    // Signed circuit inputs exercise both sign-extension and minimum values.
    let za = a128.map(|v| Z::<2>::from_twos_complement_words([v as u64, (v >> 64) as u64]));
    let zb = b128.map(|v| Z::<2>::from_twos_complement_words([v as u64, (v >> 64) as u64]));
    let zc = c256.map(|v| Z::<4>::from_twos_complement_words(*v.as_words()));
    check::<_, _, M, LANES, STEPS>(za, zb, zc, signed, signed, signed);
    let za9 = core::array::from_fn(|i| {
        Z::<9>::from_twos_complement_words(core::array::from_fn(|j| {
            if j == 8 {
                [0, 1 << 63, u64::MAX, u64::MAX >> 1][i % 4]
            } else {
                u64::MAX
            }
        }))
    });
    check::<_, _, M, LANES, STEPS>(za9, za9, za9, signed, signed, signed);
    let za5 = core::array::from_fn(|i| {
        Z::<5>::from_twos_complement_words(core::array::from_fn(|j| {
            if j == 4 {
                [0, 1 << 63, u64::MAX, u64::MAX >> 1][i % 4]
            } else {
                u64::MAX
            }
        }))
    });
    check::<_, _, M, LANES, STEPS>(za5, za5, za9, signed, signed, signed);
}
#[test]
fn all_prefix_widths_match_independent_bigint_interpolation_and_reduction() {
    width::<2, 1, 0>();
    width::<4, 3, 1>();
    width::<8, 7, 3>();
    width::<16, 15, 7>();
}
