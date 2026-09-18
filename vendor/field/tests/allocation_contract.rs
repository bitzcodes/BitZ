use field::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;
thread_local! {
    static COUNT: Cell<Option<usize>> = const { Cell::new(None) };
}
fn allocated() {
    COUNT.with(|count| {
        if let Some(n) = count.get() {
            count.set(Some(n + 1));
        }
    });
}
// This test tracks only its own thread; parallel tests and the harness cannot
// manufacture either a false pass or a failure through unrelated allocations.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        allocated();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        allocated();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        allocated();
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn caller_owned_operations_allocate_nothing() {
    let field = create_prime_field(Uint::from_words([u64::MAX - 14, (1 << 36) - 1]));
    let weights = [field.from_integer(&7u64); 17];
    let integers = [u128::MAX; 17];
    let input = [u128::MAX; 34];
    let mut out = [field.zero(); 17];
    let mut plain = [Uint::<2>::ZERO; 17];
    let mut scratch = [field.zero(); 20];
    let mut round_left = [field.from_integer(&5u64); 68];
    let mut round_right = [field.from_integer(&11u64); 68];
    let words = [u64::MAX; 9];
    let mut dot_m = [field.zero(); 34];
    let mut dot_w = [Uint::ZERO; 34];
    let binary_projection = PreparedGf128Projection::new(Gf128::new(3, 7), 32);
    let binary_input = [u64::MAX; 17];
    let mut binary_output = [Gf128::ZERO; 17];
    let signed_projection = PreparedSignedProjection::new(field.clone(), 9);
    let signed_inputs = [Z::<9>::MIN; 17];
    COUNT.with(|count| count.set(Some(0)));
    signed_projection.project_into(&signed_inputs, &mut out);
    std::hint::black_box(signed_projection.project(signed_inputs[0].as_words()));
    binary_projection.project_into::<32>(&binary_input, &mut binary_output);
    std::hint::black_box(&binary_output);
    field.batch_mul_into(&weights, &weights, &mut out);
    field.batch_invert_or_zero_ct_into(&weights, &mut out, &mut scratch);
    let claim = field.reduce(field.batch_mul_acc(&weights, &integers));
    let wide = [Uint::<32>::from_words([u64::MAX; 32]); 2];
    let pair = [weights[0], weights[1]];
    std::hint::black_box(field.weighted_pair_to_integer(&pair, &wide));
    std::hint::black_box(field.weighted_pair(&pair, &wide));
    let mut incremental = FpLinearAcc::<2, 32>::zero();
    for (weight, value) in pair.iter().zip(&wide) {
        incremental.accumulate(weight, value);
    }
    std::hint::black_box(field.reduce(incremental));
    field.fold_pairs_into(&input, &mut out, &claim);
    field.fold_pairs_into(&input, &mut plain, &claim);
    std::hint::black_box(field.from_integer(&UintRef::new(&words)));
    std::hint::black_box(field.from_integer(&ZRef::from_twos_complement_words(&words)));
    std::hint::black_box(field.eqf_single_pair_round(&round_left, &round_right, &weights, 17));
    std::hint::black_box(field.eqf_fused_fold_round(
        &mut round_left,
        &mut round_right,
        &claim,
        &weights,
        17,
    ));
    std::hint::black_box(field.eqf_grid_pass(&mut round_left, &mut round_right, &[], &weights, 17));
    std::hint::black_box(field.dot_pair_round_map(&round_left, |_| u128::MAX));
    std::hint::black_box(field.dot_fold_round_map_into(
        &round_left,
        |_| u128::MAX,
        &mut dot_m,
        &mut dot_w,
        &claim,
    ));
    std::hint::black_box((&out, &plain, &scratch, claim));
    let count = COUNT.with(|count| count.replace(None).unwrap());
    assert_eq!(count, 0);
}

#[test]
fn convenience_helpers_allocate_only_their_outputs_and_scratch() {
    let input = [Gf128::new(3, 7); 17];
    COUNT.with(|count| count.set(Some(0)));
    let products = Gf128Ops.batch_mul(&input, &input);
    let product_allocations = COUNT.with(|count| count.replace(Some(0)).unwrap());
    let inverses = Gf128Ops.batch_invert_or_zero_ct(&input);
    let inverse_allocations = COUNT.with(|count| count.replace(None).unwrap());
    assert_eq!(product_allocations, 1);
    assert_eq!(inverse_allocations, 2);
    assert_eq!(products.len(), 17);
    assert_eq!(inverses.len(), 17);
}

#[test]
fn prepared_wide_and_signed_operations_allocate_nothing() {
    let words = [u64::MAX, 0xffffffff, 0, 0xffffffff00000001];
    let field = create_prime_field(Uint::from_words(words));
    let divisor = PreparedDivisor::new(Uint::from_words(words)).unwrap();
    let inverse = PreparedOddInverse::new(Uint::from_words(words)).unwrap();
    let weights = [field.from_integer(&7u64); 8];
    let signed = [Z::<9>::MIN; 8];
    let projection = PreparedSignedProjection::new(field.clone(), 9);
    let mut projected = [field.zero(); 8];
    let mut mo = [field.zero(); 4];
    let mut wo = [Uint::ZERO; 4];
    COUNT.with(|count| count.set(Some(0)));
    projection.project_into(&signed, &mut projected);
    std::hint::black_box(&projected);
    std::hint::black_box(divisor.div_rem_ct(&Uint::<18>::MAX));
    std::hint::black_box(inverse.inverse_ct(&Uint::MAX));
    std::hint::black_box(field.dot_pair_round(&weights, &signed));
    std::hint::black_box(field.dot_fold_round_into(
        &weights,
        &signed,
        &mut mo,
        &mut wo,
        &weights[0],
    ));
    std::hint::black_box(field.dot_fold_round_into(
        &weights,
        &weights,
        &mut mo,
        &mut wo,
        &weights[0],
    ));
    let count = COUNT.with(|count| count.replace(None).unwrap());
    assert_eq!(count, 0);
}
