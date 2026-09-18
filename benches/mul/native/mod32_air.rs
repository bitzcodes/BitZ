//! Shared mod32 relation and witness for the FRI and WHIR adapters.
use super::{Corpus, Workload};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as Val;
use p3_matrix::dense::RowMajorMatrix;

pub(super) const LIMB_BASE: u64 = 1 << 16;
pub(super) const VALUE_COLUMNS: usize = 8;
pub(super) const TRACE_WIDTH: usize = VALUE_COLUMNS + 7 * 16 + 17;

#[derive(Clone, Copy)]
pub(super) struct MulAir;
impl<F> BaseAir<F> for MulAir {
    fn width(&self) -> usize {
        TRACE_WIDTH
    }
    fn max_constraint_degree(&self) -> Option<usize> {
        Some(2)
    }
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}
impl<AB: AirBuilder> Air<AB> for MulAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        // [a0, a1, b0, b1, z0, z1, c0, c1], followed by their bits.
        // c1 is 17 bits; the other seven values are 16 bits. Both carry
        // equations stay below 2^33, so a field equality is an integer equality.
        // A single a*b = z + 2^32*q equation would admit Goldilocks aliases.
        for column in 0..VALUE_COLUMNS {
            let start = VALUE_COLUMNS + 16 * column;
            let mut value = AB::Expr::ZERO;
            for bit in 0..value_bits(column) {
                let v = row[start + bit];
                builder.assert_bool(v);
                value += v.into() * AB::Expr::from_u64(1 << bit);
            }
            builder.assert_eq(row[column], value);
        }
        let base = AB::Expr::from_u64(LIMB_BASE);
        builder.assert_eq(
            row[0].into() * row[2].into(),
            row[4].into() + base.clone() * row[6].into(),
        );
        builder.assert_eq(
            row[0].into() * row[3].into() + row[1].into() * row[2].into() + row[6].into(),
            row[5].into() + base * row[7].into(),
        );
    }
}
const fn value_bits(column: usize) -> usize {
    if column == 7 { 17 } else { 16 }
}
pub(super) fn set_value(row: &mut [Val], column: usize, value: u64) {
    row[column] = Val::from_u64(value);
    for bit in 0..value_bits(column) {
        row[VALUE_COLUMNS + 16 * column + bit] = Val::from_u64((value >> bit) & 1);
    }
}
pub(super) fn generate(corpus: &Corpus) -> RowMajorMatrix<Val> {
    assert_eq!(
        corpus.workload,
        Workload::U32,
        "Plonky3 supports u32-mod32 only"
    );
    let mut values = Val::zero_vec(TRACE_WIDTH * corpus.len());
    for (row, &(a, b)) in values.chunks_exact_mut(TRACE_WIDTH).zip(corpus.inputs()) {
        assert!(a <= u32::MAX as u64 && b <= u32::MAX as u64);
        let (a0, a1, b0, b1) = (a % LIMB_BASE, a / LIMB_BASE, b % LIMB_BASE, b / LIMB_BASE);
        let low = a0 * b0;
        let c0 = low / LIMB_BASE;
        let middle = a0 * b1 + a1 * b0 + c0;
        let columns = [
            a0,
            a1,
            b0,
            b1,
            low % LIMB_BASE,
            middle % LIMB_BASE,
            c0,
            middle / LIMB_BASE,
        ];
        for (column, value) in columns.into_iter().enumerate() {
            set_value(row, column, value);
        }
    }
    RowMajorMatrix::new(values, TRACE_WIDTH)
}
pub(super) fn audit(corpus: &Corpus) -> super::WitnessAudit {
    let started = std::time::Instant::now();
    let trace = generate(corpus);
    let generation_ms = started.elapsed().as_secs_f64() * 1000.;
    let rows = trace
        .values
        .chunks_exact(TRACE_WIDTH)
        .map(|row| {
            let pair = |column: usize| {
                row[column].as_canonical_u64() + LIMB_BASE * row[column + 1].as_canonical_u64()
            };
            [pair(0), pair(2), pair(4), 0]
        })
        .collect();
    super::WitnessAudit::check(
        corpus,
        rows,
        generation_ms,
        "Plonky3 low-32-bit limb assignment",
        false,
    )
}
