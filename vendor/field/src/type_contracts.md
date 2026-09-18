The runtime context API deliberately uses an ordinary associated element type.
The caller retains the modulus and pairs values with the context that created
them; different runtime moduli of the same width are **not** different Rust types.

```
use field::*;
let field = create_prime_field(Uint::from_words([17]));
let weights = [field.from_integer(&3u64), field.from_integer(&7u64)];
let claim = field.reduce(field.batch_mul_acc(&weights, &[4u64, 5]));
assert_eq!(field.to_integer(&claim), Uint::from_words([13]));
```

Static declarations have distinct nominal identities, even with equal moduli:

```compile_fail
use field::*;
struct P; impl PrimeSpec<1> for P {const MODULUS: Uint<1> = Uint::from_words([17]);}
struct Q; impl PrimeSpec<1> for Q {const MODULUS: Uint<1> = Uint::from_words([17]);}
let p = StaticFpOps::<P,1>::new();
let q = StaticFpOps::<Q,1>::new();
let invalid = p.mul(&p.one(), &q.one());
```

The product scale is different from a field/integer linear scale:

```compile_fail
use field::*;
let field = create_prime_field(Uint::from_words([17]));
let a = [field.one()];
let invalid: FpProductAcc<1> = field.batch_mul_acc(&a, &[Uint::<1>::ONE]);
```

An accumulator from a different static prime cannot be reduced:

```compile_fail
use field::*;
struct P; impl PrimeSpec<1> for P {const MODULUS: Uint<1> = Uint::from_words([17]);}
struct Q; impl PrimeSpec<1> for Q {const MODULUS: Uint<1> = Uint::from_words([19]);}
let p = StaticFpOps::<P,1>::new();
let q = StaticFpOps::<Q,1>::new();
let invalid = q.reduce(p.batch_mul_acc(&[p.one()], &[p.one()]));
```

Wrapping integers form a ring; they do not implement field inversion:

```compile_fail
use field::*;
fn requires_field(_: impl FieldOps) {}
requires_field(WrappingOps::<Uint<2>>::new());
```

Exact integer products and carryless polynomial products are separate domains:

```compile_fail
use field::*;
let a = F2Poly::<64,1>::ZERO;
let invalid: UintProduct<1,1> = F2PolyOps.mul_wide(&a, &a);
```

Polynomial-bit construction is explicit; it cannot masquerade as numeric integer embedding.

```compile_fail
use field::Gf128;
let coefficient = Gf128::from(2u64);
```
