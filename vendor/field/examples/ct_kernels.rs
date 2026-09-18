//! Retained code-generation entrypoints for manual constant-time inspection.
//! These are probes, not a timing-security certification.
use field::*;

#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_mul(ctx: &FpCtx<2>, a: &Fp<2>, b: &Fp<2>) -> Fp<2> {
    ctx.mul(a, b)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_linear(ctx: &FpCtx<2>, a: &[Fp<2>], b: &[u64]) -> Fp<2> {
    ctx.reduce(ctx.batch_mul_acc(a, b))
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_divrem(a: &Uint<2>, b: &Uint<2>) -> CtValue<(Uint<2>, Uint<2>)> {
    a.checked_div_rem_ct(b)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_gf8(a: Gf8, b: Gf8) -> Gf8 {
    Gf8Ops.mul(&a, &b)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_embedding(a: Gf8) -> Gf128 {
    Gf128Ops.embed(&a)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_pow(ctx: &FpCtx<2>, a: &Fp<2>, e: &Uint<2>) -> Fp<2> {
    ctx.pow_ct(a, e)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_batch_mul(ctx: &FpCtx<2>, a: &[Fp<2>], b: &[Fp<2>], out: &mut [Fp<2>]) {
    ctx.batch_mul_into(a, b, out)
}
#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_borrowed_signed(ctx: &FpCtx<2>, words: &[u64]) -> Fp<2> {
    ctx.from_integer(&ZRef::from_twos_complement_words(words))
}
#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_native_round(ctx: &FpCtx<2>, weights: &[Fp<2>], values: &[u64]) -> [Fp<2>; 2] {
    ctx.dot_pair_round(weights, values)
}
#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_b127_square(value: B127, n: usize) -> B127 {
    value.square_n(n)
}
#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_b127_mul(a: &B127, b: &B127) -> B127 {
    B127Ops.mul(a, b)
}
#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_gf128_prepared(ctx: &PreparedGf128Mul, value: &Gf128) -> Gf128 {
    ctx.mul(value)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_prepared_division(
    ctx: &PreparedDivisor<4>,
    value: &Uint<9>,
) -> (Uint<9>, Uint<4>) {
    ctx.div_rem_ct(value)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_prepared_inverse(
    ctx: &PreparedOddInverse<4>,
    value: &Uint<4>,
) -> CtValue<Uint<4>> {
    ctx.inverse_ct(value)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_signed_projection(ctx: &PreparedSignedProjection<2>, words: &[u64]) -> Fp<2> {
    ctx.project(words)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn field_probe_canonical_mul(ctx: &FpCtx<2>, value: &Uint<2>, coefficient: &Fp<2>) -> Uint<2> {
    ctx.mul_canonical(value, coefficient)
}

fn main() {
    let ctx = create_prime_field(Uint::from_words([u64::MAX - 14, (1 << 36) - 1]));
    let a = ctx.from_integer(&std::hint::black_box(7u64));
    std::hint::black_box(field_probe_mul(&ctx, &a, &a));
    std::hint::black_box(field_probe_linear(&ctx, &[a; 2], &[3, 5]));
    std::hint::black_box(field_probe_divrem(&Uint::MAX, &Uint::from_words([17, 0])));
    std::hint::black_box(field_probe_gf8(Gf8(3), Gf8(5)));
    std::hint::black_box(field_probe_embedding(Gf8(128)));
    std::hint::black_box(field_probe_pow(&ctx, &a, &Uint::from_words([17, 0])));
    field_probe_batch_mul(&ctx, &[a; 2], &[a; 2], &mut [a; 2]);
    std::hint::black_box(field_probe_borrowed_signed(&ctx, &[u64::MAX; 4]));
    std::hint::black_box(field_probe_native_round(&ctx, &[a; 2], &[0, u64::MAX]));
    let binary = B127::from_polynomial_words([u64::MAX, u64::MAX >> 1]);
    std::hint::black_box(field_probe_b127_square(binary, 17));
    std::hint::black_box(field_probe_b127_mul(&binary, &binary));
    std::hint::black_box(field_probe_gf128_prepared(
        &PreparedGf128Mul::new(Gf128::new(7, 13)),
        &Gf128::new(11, 17),
    ));
    let modulus = Uint::from_words([u64::MAX, 0xffffffff, 0, 0xffffffff00000001]);
    let divisor = PreparedDivisor::new(modulus).unwrap();
    let inverse = PreparedOddInverse::new(modulus).unwrap();
    std::hint::black_box(field_probe_prepared_division(&divisor, &Uint::MAX));
    std::hint::black_box(field_probe_prepared_inverse(&inverse, &Uint::MAX));
}
