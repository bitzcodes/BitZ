//! Delayed-reduction multiply-accumulate for sumcheck inner loops.
//!
//! The eq-factored sumcheck's round body accumulates many products whose
//! results feed only field *additions* (the round-polynomial coefficient
//! sums), never another multiplication. For fields whose reduction step is
//! a substantial fraction of the multiply — `GF(2^128)` via carryless
//! Karatsuba, where reduction mod `X^128 + X^7 + X^2 + X + 1` costs two
//! word-times-g folds — those products can be accumulated in their
//! **unreduced** wide form (XOR of 256-bit carryless products) and reduced
//! once per accumulator per round. Reduction is `F_2`-linear, so
//! `reduce(Σ wide_i) = Σ reduce(wide_i)` exactly: the final field elements
//! (and every transcript byte downstream) are unchanged.
//!
//! Prime fields take the trivial instance (`Wide = Self`, reduced ops), so
//! the driver stays field-generic with zero overhead there.

/// Multiply-accumulate with an opaque (possibly unreduced) accumulator.
///
/// Laws (all exact, no approximation):
/// - `from_wide(wide_zero(zero)) == zero`
/// - `from_wide(w) == x + y` when `w = wide_add(wide_of(x), wide_of(y))`
/// - `from_wide(mul_wide(a, b)) == a * b`
///
/// so any sum of products computed through this trait equals the same sum
/// computed with reduced multiplies — bit-for-bit.
pub trait WideMulAcc: Sized {
    /// Optional inverse for a public equality coordinate. Unsupported fields
    /// and zero return `None`, retaining the three-coefficient GKR kernel.
    fn eqf_inverse(&self) -> Option<Self> {
        None
    }

    /// The accumulator representation (unreduced for char-2 carryless
    /// fields; `Self` for fields whose multiply is cheapest reduced).
    type Wide: Clone + Send;

    /// The additive-identity accumulator. Takes the field zero so
    /// runtime-config fields can seed config-carrying values.
    fn wide_zero(zero: &Self) -> Self::Wide;

    /// The wide form of an already-reduced field element.
    fn wide_of(x: &Self) -> Self::Wide;

    /// `a · b` in wide form (reduction deferred).
    fn mul_wide(a: &Self, b: &Self) -> Self::Wide;

    /// `acc += x`.
    fn wide_add_assign(acc: &mut Self::Wide, x: &Self::Wide);

    /// `acc -= x`.
    fn wide_sub_assign(acc: &mut Self::Wide, x: &Self::Wide);

    /// Reduce the accumulator to a field element.
    fn from_wide(w: Self::Wide) -> Self;

    /// Conditional accumulate `if mask { *acc += x }` — override with a
    /// BRANCHLESS select where the representation allows (char-2 fields:
    /// `acc ^= (0−mask) & x`), so hot loops driven by unpredictable data
    /// bits pay a constant ~1 ns instead of a coin-flip branch.
    /// Value-exact either way.
    fn add_assign_masked(acc: &mut Self, x: &Self, mask: bool);

    /// Optional fused kernel for the eq-factored sumcheck's single-pair
    /// round body: over `b < half`, with `l0 = l[2b]`, `l1 = l[2b+1]`
    /// (same for `r`), returns
    /// `Some((Σ w_b·l0·r0, Σ (w_b·l1·r1 − w_b·l0·r0 − w_b·Δl·Δr), Σ w_b·Δl·Δr))`
    /// where `Δl = l1 − l0` weighted as `w·l1 − w·l0`. Any override must be
    /// VALUE-EXACT (only reorder exact field ops); return `None` to use the
    /// driver's generic loop. Lets a field ship a hand-scheduled kernel
    /// (e.g. interleaved PMULL chains) without the driver losing genericity.
    fn eqf_single_pair_round(
        _l: &[Self],
        _r: &[Self],
        _w: &[Self],
        _half: usize,
    ) -> Option<(Self, Self, Self)> {
        None
    }

    /// Optional fused kernel for the eq-factored sumcheck's TWO-pair round
    /// body (the fraction-GKR layer combine `L₀·R₀ + L₁·R₁` per slot —
    /// `quotient_gkr`'s `nl·dr + dl·aux`): over `b < half`, with each
    /// pair's entries at `2b, 2b+1` and the shared weight `w_b`, returns
    /// the three round-polynomial coefficients
    /// `Some((Σ_b Σ_p w_b·l0·r0, Σ_b Σ_p (w_b·l1·r1 − w_b·l0·r0 − w_b·Δl·Δr), Σ_b Σ_p w_b·Δl·Δr))`.
    /// Any override must be VALUE-EXACT vs the driver's generic multi-pair
    /// loop (only reorder exact field ops — for char-2 wide accumulation,
    /// `w·(A+B) = w·A + w·B` and reduction is `F₂`-linear); return `None`
    /// to use the generic loop.
    fn eqf_two_pair_round(
        _l0: &[Self],
        _r0: &[Self],
        _l1: &[Self],
        _r1: &[Self],
        _w: &[Self],
        _half: usize,
    ) -> Option<(Self, Self, Self)> {
        None
    }

    /// Optional fused kernel for the eq-factored fold
    /// `v[b] ← v[2b] + ρ·(v[2b+1] − v[2b])` for `b < half` (caller
    /// truncates). Must be value-exact; return `false` to use the generic
    /// in-place loop.
    fn eqf_fold_in_place(_v: &mut [Self], _rho: &Self, _half: usize) -> bool {
        false
    }

    /// Optional fused kernel for the pass-fusion path (`BITZ_EQF_FUSE`): the
    /// DEFERRED fold of the previous round fused with this round's
    /// single-pair message body, in one pass. `l`/`r` hold `4·half`
    /// unfolded entries; for `b < half` the kernel folds
    /// `fl0 = l[4b] + ρ·(l[4b+1] − l[4b])`,
    /// `fl1 = l[4b+2] + ρ·(l[4b+3] − l[4b+2])` (same for `r`), writes them
    /// to the buffer prefix (`l[2b] = fl0`, `l[2b+1] = fl1` — writes trail
    /// the reads, so in place is safe; the caller truncates to `2·half`),
    /// and accumulates this round's coefficients over the folded pairs:
    /// `(Σ w_b·fl0·fr0, Σ (w_b·fl1·fr1 − w_b·fl0·fr0 − w_b·Δl·Δr), Σ w_b·Δl·Δr)`
    /// with `Δl = w·fl1 − w·fl0`, `Δr = fr1 − fr0`. Any override must be
    /// VALUE-EXACT vs folding with [`Self::eqf_fold_in_place`]'s formula
    /// and then running [`Self::eqf_single_pair_round`]'s body on the
    /// folded buffers; return `None` to use the driver's generic fused
    /// loop.
    fn eqf_fused_fold_round(
        _l: &mut [Self],
        _r: &mut [Self],
        _rho: &Self,
        _w: &[Self],
        _half: usize,
    ) -> Option<(Self, Self, Self)> {
        None
    }

    /// Optional kernel for the double-fold dense grid pass
    /// (`BITZ_EQF_DOUBLE`'s `dense_grid_pass` body): per quad `b < quads`,
    /// fold the `pending` deferred challenges (`d = pending.len() ≤ 2`)
    /// into the 4 logical values `lv[i] = fold(l, (b≪2)|i, pending)` (same
    /// for `r`), write them back to the buffer prefix when `d > 0` (the
    /// caller truncates), weight the `L` side by `suffix[b]`, and
    /// accumulate the two 3×3 node grids' nine products. Returns the
    /// `X₁`-monomial-converted nine coefficients (`[A_u[v]]` at `u·3+v`) —
    /// exactly the generic pass's return. Any override must be VALUE-EXACT
    /// (only reorder exact field ops; for the arity-4 fold, the expansion
    /// `v₀ + ρ₁·(v₁−v₀) + ρ₂·(v₂−v₀) + ρ₁ρ₂·(v₃−v₂−v₁+v₀)` and shared
    /// reduction are exact — reduction is `F₂`-linear); return `None` to
    /// use the driver's generic pass.
    fn eqf_grid_pass(
        _l: &mut [Self],
        _r: &mut [Self],
        _pending: &[Self],
        _suffix: &[Self],
        _quads: usize,
    ) -> Option<[Self; 9]> {
        None
    }
}
