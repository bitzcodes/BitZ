# Shared arithmetic API

This describes the implemented core. [IMPLEMENTATION.md](IMPLEMENTATION.md) tracks integration and performance work still in progress; [PLAN.md](PLAN.md) retains the full migration and acceptance plan. Scalar construction and accumulator payloads are controlled by the shared crate. Low-level architecture entry points are exposed for fused consumers; their complete target-feature gates are mandatory. There are no legacy scalar compatibility aliases.

## Contexts and values

```rust
use field::*;

let field = create_prime_field(Uint::from_words([17]));
let a = field.from_integer(&3u64);
let b = field.from_integer(&7u64);
let c = field.mul(&a, &b);
let weights = vec![a, b, c];
let integers = [4u64, 5, 6];
let acc = field.batch_mul_acc(&weights, &integers);
let claim = field.reduce(acc);
assert_eq!(field.to_integer(&claim), Uint::from_words([3]));
```

| Type | Meaning |
|---|---|
| `Uint<L>` | Exactly L little-endian u64 limbs, unsigned, L > 0. |
| `Z<L>` | Exactly L limbs, signed two's complement. |
| `UintRef<'a>`, `ZRef<'a>` | Borrowed nonempty limbs of a public declared width. |
| `Bit` | An integer-valued bit. |
| `CtMask`, `CtValue<T>` | Opaque mask and a value accompanied by masked validity. |
| `FpCtx<L>` | One immutable runtime modulus and prepared arithmetic constants. |
| `Fp<L>` | Compact Montgomery value, exactly 8L bytes, without a context pointer. |
| `StaticFpOps<P,L>` | Zero-size context for a nominal static prime specification. |
| `StaticFp<P,L>` | Same compact Montgomery storage, with P in its type. |
| `RuntimePrime`, `StaticPrime<P>` | Type-level identities used by `PrimeValue<Id,L>`. |
| `ModRingCtx<L>`, `Residue<L>` | Arbitrary public modulus >1 and canonical residue storage. |
| `IntegerOps`, `WrappingOps<T>` | Exact integer operations and explicit arithmetic modulo storage width. |
| `F2`, `Gf8`, `Gf128`, `B127` | Binary fields with matching F2Ops/Gf8Ops/Gf128Ops/B127Ops providers. |
| `F2Poly<BITS,WORDS>`, `F2PolyOps` | Exact bounded binary polynomials, separate from field values. |

`FpCtx<L>: RingOps<Elem = Fp<L>>`. An associated type belongs to the **provider type**, not to an individual runtime value. Two runtime primes with the same L have the same Rust element type. Using an element with its originating modulus is the caller's contract, including codecs, prepared values, and accumulators. There are no lifetime brands, scopes, context buffers, per-element identity tags, or release identity checks.

Static prime identities are different Rust types, even if two specifications contain the same modulus. All scaled products and accumulators carry that identity too. Standard `+`, `-`, `*`, and assignment operators are available for static elements; runtime elements use context methods.

The only new prime-field scalar representation is Montgomery: x is stored as xR mod p, R = 2^(64L). There is no FpPlain or FpSmall. Canonical folded values use `Uint<L>` or appropriately wide native integer storage. The owning consumer retains the context when those integers represent cached residues. An ordinary integer operand need not be below p and can be embedded under any modulus.

`create_prime_field(prime)` trusts an odd prime >=3; debug builds check it, release builds do not test primality. Public proof decoding and public slice-shape checks remain checked in release. This constructor policy is not a reason to accept noncanonical proof bytes.

## Scalar traits

```rust
pub trait RingOps {
    type Elem: Copy + Send + Sync + CtEq + CtSelect;
    fn zero(&self) -> Self::Elem;
    fn one(&self) -> Self::Elem;
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    fn neg(&self, a: &Self::Elem) -> Self::Elem;
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    fn square(&self, a: &Self::Elem) -> Self::Elem;
    fn pow_ct<const E: usize>(&self, base: &Self::Elem, exponent: &Uint<E>) -> Self::Elem;
}
pub trait FieldOps: RingOps {
    fn inverse_ct(&self, a: &Self::Elem) -> CtValue<Self::Elem>;
    fn div_ct(&self, a: &Self::Elem, b: &Self::Elem) -> CtValue<Self::Elem>;
}
pub trait IntegerEmbedding<T>: RingOps {
    fn from_integer(&self, value: &T) -> Self::Elem;
}
pub trait FieldEmbedding<T>: FieldOps {
    fn embed(&self, value: &T) -> Self::Elem;
}
```

`&C` forwards RingOps, FieldOps, and IntegerEmbedding to C, allowing preparation to borrow an existing context. IntegerOps itself is not a ring: exact bounded multiplication has a wider output. `WrappingOps<Uint<L>>` and `WrappingOps<Z<L>>` are rings, not fields. ModRingCtx exposes an inherent `inverse_ct` that returns invalid for every nonunit, including zero, and does not implement FieldOps.

In characteristic two, **numeric integer embedding is parity**. `Gf8(byte)` constructs a polynomial-basis byte; `Gf128Ops.embed(&byte)` uses the AES subfield embedding. These are distinct operations. `Gf128::new(lo,hi)` and `Gf128::from_polynomial_bits(u128)` explicitly construct polynomial coefficients in GHASH order. There is no ambiguous `From<u64/u128>` numeric conversion.

## Exact products, MAC, and reduction

```rust
pub trait WideMul<Lhs, Rhs = Lhs> {
    type Product;
    fn mul_wide(&self, lhs: &Lhs, rhs: &Rhs) -> Self::Product;
}
pub trait BatchMulAcc<Lhs, Rhs = Lhs> {
    type Accumulator;
    fn batch_mul_acc(&self, lhs: &[Lhs], rhs: &[Rhs]) -> Self::Accumulator;
    fn batch_mul_acc_map(&self, len: usize,
        term: impl FnMut(usize) -> (Lhs, Rhs)) -> Self::Accumulator;
}
pub trait MergeAccumulator: Sized {
    fn zero() -> Self;
    fn merge_assign(&mut self, rhs: &Self);
}
pub trait Reduce<Input> {
    type Output;
    fn reduce(&self, input: Input) -> Self::Output;
}
```

Indexed MAC calls its callback exactly once at every ascending index. Callers own parallelism and merge partial accumulators. All exact integer sums have one headroom limb and support **fewer than 2^64 full-width terms in total**, including merged batches. A single 64-bit slice batch fits that bound; arbitrarily repeated merges retain the caller contract. There is no count field, release overflow check, per-term try_mac, or hidden allocation. Shape checks precede the loop; arithmetic bounds have debug assertions. Invalid public shapes panic unless a preparation constructor returns ShapeError.

| Operands/provider | Exact product | Accumulator |
|---|---|---|
| Uint<A> x Uint<B>, IntegerOps | UintProduct<A,B> | UintAccumulator<A,B> |
| Signed or mixed Uint/Z, IntegerOps | ZProduct<A,B> | ZAccumulator<A,B> |
| u32/u64/u128 x u32/u64/u128, IntegerOps | UintProduct<words(lhs),words(rhs)> | Corresponding UintAccumulator |
| Fp<L> x Fp<L>, FpCtx<L> | FpProduct<L>, scale R² | FpProductAcc<L>, scale R² |
| Fp<L> x Uint<N>/native integer | FpLinearProduct<L,N>, scale R | FpLinearAcc<L,N>, scale R |
| Fp<L> x Z<N> | FpSignedLinearProduct<L,N>, scale R | FpSignedLinearAcc<L,N>, scale R |
| StaticFp<P,L> combinations | StaticFpProduct/LinearProduct/SignedLinearProduct | Corresponding StaticFp*Acc |
| Gf128 x Gf128 or Gf8 | Gf128Product | XorAccumulator<Gf128Product> |
| Gf8 x Gf8 | Gf8Product | XorAccumulator<Gf8Product> |
| B127 x B127 | B127Product | XorAccumulator<B127Product> |
| F2 x F2 | F2 | XorAccumulator<F2> |
| F2Poly<_,A> x F2Poly<_,B> | F2PolyProduct<A,B> | XorAccumulator<F2PolyProduct<A,B>> |

Native u32/u64/Bit occupy one word and u128 occupies two words in prime mixed products. Native word assembly happens inside each consuming operation; no converted operand vector is allocated.

Prime and modular contexts also accept `UintRef` and `ZRef` through
`IntegerEmbedding`. Their public slice length fixes the work; signed projection
uses a masked magnitude view without allocating a temporary word vector. Exact
MAC retains fixed operand dimensions; use `Uint<N>` in an indexed reader for
segmented inputs whose public width is known at compile time.

```rust
pub type PrimeProductAcc<Id, const L: usize> =
    ScaledAccumulator<Id, UintAccumulator<L,L>, MontyR2>;
pub type PrimeLinearAcc<Id, const L: usize, const N: usize> =
    ScaledAccumulator<Id, UintAccumulator<L,N>, MontyR>;
pub type PrimeSignedLinearAcc<Id, const L: usize, const N: usize> =
    ScaledAccumulator<Id, ZAccumulator<L,N>, MontyR>;
```

The runtime aliases substitute RuntimePrime for Id; static aliases substitute StaticPrime<P>. Matching ScaledProduct aliases wrap UintProduct/ZProduct instead. Scale R² reduction removes one Montgomery factor; scale R reduction takes a remainder and retains R. Reducing an unscaled exact integer product/accumulator through a prime context embeds the result into Montgomery form. Signed reduction handles the magnitude and sign without branching on the sign.

## Integers and masks

```rust
impl<const L: usize> Uint<L> {
    pub const fn from_words(words: [u64;L]) -> Self;
    pub const fn as_words(&self) -> &[u64;L];
    pub fn wrapping_add(&self, rhs: &Self) -> Self;
    pub fn wrapping_sub(&self, rhs: &Self) -> Self;
    pub fn wrapping_mul(&self, rhs: &Self) -> Self;
    pub fn wrapping_neg(&self) -> Self;
    pub fn zero_extend<const M: usize>(&self) -> Uint<M>;
    pub fn checked_resize_ct<const M: usize>(&self) -> CtValue<Uint<M>>;
    pub fn checked_to_signed_ct(&self) -> CtValue<Z<L>>;
    pub fn bit(&self, public_index: usize) -> Bit;
}
impl<const L: usize> Z<L> {
    pub const fn from_twos_complement_words(words: [u64;L]) -> Self;
    pub const fn as_words(&self) -> &[u64;L];
    // Same inherent wrapping operations as Uint, including wrapping_neg.
    pub fn sign_extend<const M: usize>(&self) -> Z<M>;
    pub fn checked_resize_ct<const M: usize>(&self) -> CtValue<Z<M>>;
    pub fn checked_to_unsigned_ct(&self) -> CtValue<Uint<L>>;
    pub fn is_negative_ct(&self) -> CtMask;
    pub fn unsigned_abs(&self) -> Uint<L>;
}
pub trait CheckedArithmetic: Sized {
    fn checked_add_ct(&self, rhs: &Self) -> CtValue<Self>;
    fn checked_sub_ct(&self, rhs: &Self) -> CtValue<Self>;
    fn checked_mul_ct(&self, rhs: &Self) -> CtValue<Self>;
    fn checked_neg_ct(&self) -> CtValue<Self>;
    fn checked_div_rem_ct(&self, rhs: &Self) -> CtValue<(Self,Self)>;
}
```

Uint/Z implement checked and wrapping traits, bitwise operations, public-index shifts, CtEq/CtOrd/CtSelect. Checked division masks zero divisors and signed MIN/-1. Extension dimensions are checked at compile time. Exact products/accumulators expose limb parts and checked resizing rather than silently discarding high limbs. CtValue exposes value(), validity(), into_parts(), map(), and and_then(); declassification is explicit through CtMask::declassify(). Ordinary PartialEq/Debug and legacy Option-returning arithmetic are not the masked private-input API.

Native `From<u64>`/`From<i128>` construction of `Z<L>` is exact and panics if
a one-limb destination cannot represent the value. To retain masked failure,
construct in `Z<2>` and call `checked_resize_ct::<1>()`, or use
`Uint<1>::checked_to_signed_ct()`. Intentional bit-pattern reinterpretation uses
`from_twos_complement_words`; it is separate from native integer construction.

## Batch helpers and folding

```rust
pub trait BatchFieldOps: FieldOps {
    fn batch_mul_into(&self, lhs: &[Self::Elem], rhs: &[Self::Elem], out: &mut [Self::Elem]);
    fn batch_mul(&self, lhs: &[Self::Elem], rhs: &[Self::Elem]) -> Vec<Self::Elem>;
    fn batch_invert_or_zero_ct_into(&self, input: &[Self::Elem],
        out: &mut [Self::Elem], scratch: &mut [Self::Elem]);
    fn batch_invert_or_zero_ct(&self, input: &[Self::Elem]) -> Vec<Self::Elem>;
}
pub trait FoldPairs<Src,Dst>: RingOps {
    fn fold_pairs_into(&self, src: &[Src],
        dst: &mut [Dst], challenge: &Self::Elem);
    fn fold_pairs_map_into(&self,
        read: impl FnMut(usize) -> Src, dst: &mut [Dst], challenge: &Self::Elem);
}
```

The pair-fold, batch-multiply, and batch-invert `_into` methods allocate nothing. Batch multiplication's output length equals both inputs; inversion requires output length n and scratch length at least n. Zero entries stay zero. The allocating helpers allocate one output, or output plus scratch respectively. Inputs/output/scratch are distinct safe Rust borrows.

Prime folding writes `(1-r)*src[2j] + r*src[2j+1]`, j=i. Native Bit/u32/u64/u128/Uint<N> are consumed by mixed MAC; Z<N> and Fp input variants are also supported. Dst can be Fp or canonical Uint<L>. Indexed reads visit each selected source once in order and can assemble segmented u128/u256 storage without intermediate vectors. The source length must equal twice the output length; mapped folds derive that extent from the output. Range arithmetic is checked before the loop.

`fold_prefix_into(src, dst, challenges)` and `fold_prefix_map_into(read, dst, challenges)` fold already-known Boolean challenges together: `dst[i] = sum_b eq(challenges,b) src[2^k*i+b]`, with challenge zero binding the low bit. The default implementation allocates `2^k` equality weights and uses mixed batch MACs, including for native integer inputs. It reads each source entry once. This is not the Lagrange fold used by a univariate skip.

Prime contexts expose `fold_in_place(&mut [Fp], &challenge, half)` and `fold_plain_in_place(&mut [Uint<L>], &challenge, half)`, writing only the prefix. The latter takes already-canonical values in [0,p); debug assertions check that caller contract. Unlike the generic Uint fold, it avoids re-projecting canonical inputs. For a native width narrower than the field width, canonical-output folding uses one Montgomery reduction under an algebraically established bound.

SumcheckKernels is implemented for prime and binary providers. It exposes `eqf_single_pair_round`, `eqf_two_pair_round`, `eqf_fold_in_place`, `eqf_fused_fold_round`, and `eqf_grid_pass`. Weighted pair rounds return `[constant,linear,quadratic]`; the grid returns nine entries in the existing X1-monomial/X2-node order. Fused methods traverse and write the prefix directly, retain delayed coefficient accumulation, and reduce weighted intermediate products before multiplying again. Fixed multipliers are prepared once per fold pass, including the retained ARM GF128 preparation. Selected unrolled/vector schedules and consumer performance qualification remain pending.

Runtime and static prime providers implement these native inner-product kernels
for Src = Bit/u32/u64/u128/Uint<N>/Z<N>/their own field element and Folded = Uint<L>:

```rust
pub trait DotProductKernels<Src,Folded>: FieldOps {
    fn dot_pair_round(&self, weights:&[Self::Elem], values:&[Src]) -> [Self::Elem;2];
    fn dot_pair_round_map(&self, weights:&[Self::Elem],
        read:impl FnMut(usize)->Src) -> [Self::Elem;2];
    fn dot_fold_round_into(&self, weights:&[Self::Elem], values:&[Src],
        weights_out:&mut[Self::Elem], values_out:&mut[Folded],
        challenge:&Self::Elem) -> [Self::Elem;2];
    fn dot_fold_round_map_into(&self, weights:&[Self::Elem],
        read:impl FnMut(usize)->Src, weights_out:&mut[Self::Elem],
        values_out:&mut[Folded], challenge:&Self::Elem) -> [Self::Elem;2];
}
```

The message is `[constant,quadratic]`; the current claim determines the linear
coefficient. Each reader is called once per input, in ascending order. Full-width
integer differences use a masked sign and magnitude, so u64/u128 maxima do not
overflow a same-width signed temporary. A fused fold writes Montgomery weights
and canonical integer witnesses, then computes the next message in that same
traversal. The inherent `dot_fold_plain_round_into` specializes later rounds
whose witness inputs are already canonical, avoiding another projection. Public
shape errors are checked before reading callbacks or changing outputs.

Canonical-through-Wengert operations are explicit context methods:

```rust
fn mul_canonical(&self, value: &Uint<L>, coefficient: &Self::Elem) -> Uint<L>;
fn mul_canonical_into(&self, values: &[Uint<L>], coefficients: &[Self::Elem], out: &mut [Uint<L>]);
fn reduce_linear_to_integer<const N: usize>(&self, input: PrimeLinearAcc<Id,L,N>) -> Uint<L>;
```

The multiplication requires an already-canonical input and uses one Montgomery
reduction directly; no field-table projection is introduced. Ordinary `reduce`
continues returning the Montgomery element for the same accumulator. Signed
first folds and dot rounds consume the signed integers directly, without forming
an overflowing same-width difference or projecting a whole table.

## Preparation, sampling, and encoding

| Type/trait | Important methods |
|---|---|
| PreparedDivisor<D> | new(Uint<D>) -> Result; prepared radix Barrett division via div_rem_ct<N>(&Uint<N>) and div_rem_product_ct<A,B>(&UintProduct<A,B>). |
| PreparedOddInverse<L> | new(Uint<L>) -> Result; inverse_ct(&Uint<L>) -> CtValue<Uint<L>>; fixed 128L-step binary extended GCD, including composite odd moduli. |
| FixedBasePow<C,E> | new(C,C::Elem); new_public(C,C::Elem,window); pow_ct(&Uint<E>) and explicit public-exponent-only pow_public(&Uint<E>). |
| PreparedProducts<'a,A,B> | new(lhs,rhs,out,PublicProductBounds) -> Result; execute(); outputs(). |
| PublicProductBounds | Public lhs_limbs/rhs_limbs. Values must fit those declared widths. |
| PreparedIntegerProjection<C> | new(C); project(&T); project_into(&[T],&mut[C::Elem]). |
| PreparedGf128Mul / PreparedB127Mul | new(scalar), mul(&value), mul_into(input,out); ARM GF128 retains its interleaved fixed-multiplier preparation. |
| CanonicalCodec<T> | encoded_len(); encode_into(&T,&mut[u8]); decode_ct(&[u8]) -> Result<CtValue<T>,DecodeError>; decode_public(). |
| FieldSampling | sample_public(source,max_draws); sample_nonzero_public(source,max_draws). |
| PublicRandomSource | fill_bytes(&mut [u8]); caller supplies an appropriate public RNG/transcript adapter. |
| PrimeSearchPolicy | target_security_bits:u32, max_candidates:u64, max_rejection_draws:usize. |
| ProbablePrime<L> | Accepted candidate plus prepared parameters; modulus(). |
| PrimeSearchError, SamplingError, ContextError, ShapeError, DecodeError | Explicit public setup/shape/codec failures. |

`sample_prime_public(source, interval, &policy)` returns ProbablePrime<L>; `FpCtx::from_prime` reuses its constants. FpCtx::sample_prime_public combines the two. Sampling is variable-time and public-only. Candidate/base draws use bounded rejection; successful prime outputs are not claimed to be exactly uniform. PrimeSearchPolicy::minimum_rejection_draws() derives a mandatory whole-search rejection-exhaustion bound separately from the randomized Miller–Rabin composite-acceptance budget. Weak policies fail before consuming randomness. It is not run at compile time. Static declarations use `define_prime_field!`, which generates a PrimeSpec, element/context aliases, and a runtime CI primality test. Const evaluation only prepares width-bounded arithmetic constants.

Codecs use fixed little-endian words, reject noncanonical residues/padding, and keep Montgomery words private. GF128 serde preserves Flock's lo/hi structure when the optional serde feature is enabled. Spongefish codecs cover Gf128 and static prime types; runtime prime codecs remain context methods. Legacy Fq/F128 numeric APIs are removed. Q100Prime/Q100Element/Q100Field declare the existing 2^100 - 15 prime.

## Coherence and performance contract

1. RingOps owns Elem; arithmetic providers have concrete implementations.
2. BatchMulAcc<Lhs,Rhs> has one accumulator for that operand pair. No overlapping blanket `impl<T: Into<...>>` or specialization requirement.
3. Static identity and Montgomery scale are type parameters in products and accumulators. Runtime modulus association remains a caller contract.
4. Reduce<Input> has one output for each provider/input pair. FoldPairs<Src,Dst> deliberately includes output representation in its trait arguments.
5. Exact integer, wrapping integer, carryless polynomial, binary-field product, R, and R² domains stay distinct.
6. Native-input loops use declared widths and public shape, never private magnitude/zero-based dispatch. Callbacks must uphold that timing contract too.
7. Correctness, absence of allocations, and optimized instruction schedules do not themselves establish throughput. Actual API and consumer gates are separate, on ARM and x86, with retained references and per-case results.

## Explicit representation and prepared signed projection

```rust
// Runtime and static providers both expose these boundaries.
fn from_montgomery_integer(&self, value: Uint<L>) -> Self::Elem;
fn to_integer(&self, value: &Self::Elem) -> Uint<L>;
fn reduce_integer<const N: usize>(&self, value: &Uint<N>) -> Uint<L>;
fn mul_canonical(&self, value: &Uint<L>, coefficient: &Self::Elem) -> Uint<L>;
fn mul_canonical_into(&self, values: &[Uint<L>], coefficients: &[Self::Elem], out: &mut [Uint<L>]);

impl<const L: usize> PreparedSignedProjection<L> {
    pub fn new(field: FpCtx<L>, max_words: usize) -> Self;
    pub fn project_canonical(&self, words: &[u64]) -> Uint<L>;
    pub fn project(&self, words: &[u64]) -> Fp<L>;
    pub fn project_into<const N: usize>(&self, input: &[Z<N>], out: &mut [Fp<L>]);
}
```

`PrimeValue::as_montgomery_integer` exposes the stored encoding explicitly.
`from_montgomery_integer` trusts the originating context, scale and reduced range;
its range assertion is debug-only. Untrusted bytes use canonical decoding.
`mul_canonical` requires a canonical input below the modulus and leaves its result
canonical. It does not create a second field-element representation type.

Prepared signed projection retains public radix/sign-correction powers. Input
length is a public declared width, including sign limbs. The `_into` method
checks lengths/capacity once and allocates nothing. `Z<L>` standard arithmetic
operators implement the wrapping ring; use checked methods or exact products
when overflow must be rejected rather than wrapped.


## Incremental kernels and canonical boundaries

`FpProductAcc<L>::accumulate(&Fp<L>, &Fp<L>)`,
`FpLinearAcc<L,N>::accumulate(&Fp<L>, &Uint<N>)`, and
`FpSignedLinearAcc<L,N>::accumulate(&Fp<L>, &Z<N>)` support fused consumer kernels.
The owner validates public batch capacity before the loop, including merged
batches (fewer than 2^64 full-width terms). These calls allocate nothing and do not
perform a per-term capacity check. `merge_assign`, `AddAssign`, and `Add` retain
the same field identity and scale. `unreduced_integer()` gives read-only access;
it cannot turn a product accumulator into a linear accumulator.

Prime providers also expose:

```rust
fn from_canonical_integer(&self, value: &Uint<L>) -> Self::Elem;
fn weighted_pair<const N: usize>(&self, coefficients: &[Self::Elem;2], values: &[Uint<N>;2]) -> Self::Elem;
fn weighted_pair_to_integer<const N: usize>(&self, coefficients: &[Self::Elem;2], values: &[Uint<N>;2]) -> Uint<L>;
```

The first method requires a canonical input (debug-checked). Weighted pairs accept
full-width arbitrary integers, fuse their embedding into the multiply, and name
the output representation explicitly. For `FpCtx<2>`, canonical-u128 helpers
support existing integer-owned tables; `mul_canonical_u128` takes a prepared
Montgomery factor. `reduce_montgomery_bounded(&Uint<4>) -> Uint<2>` requires the
owner to establish `input < modulus * 2^128` and returns canonical integer output.

Binary product domains expose `from_element`, `zero`, XOR combination and reduction.
`B127Product::from_product_words` requires degree <=252, while
`Gf128Product::from_polynomial_words` accepts any 256-bit polynomial. The types
remain distinct and their reducers cannot be interchanged. `B127` offers both a
trusted `from_canonical_words` constructor and masked `from_canonical_ct` decoding.

## Shared binary projections and consumer ownership

`PreparedGf128Projection::new(point, bits)` prepares powers for up to 64 bits.
`project(bits: u64) -> Gf128` consumes one packed polynomial, and
`project_into::<D>(&[u64], &mut [Gf128])` uses the retained four-lane NEON kernel
with a fixed-bound scalar tail. Construction allocates the powers; `_into`
allocates nothing. High input bits outside the declared width are ignored.

BitZ and Flock use the same `Gf128` type and canonical codec. Runtime proof owners
retain one `FpCtx<2>` for their `Fp<2>` values; codecs require that context rather
than reconstructing it from each element. `XorAccumulator::into_payload` exposes
an unreduced binary product for explicit output representation conversion.
