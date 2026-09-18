# Field and integer arithmetic

The shared package lives here under the Cargo package name `field`. Its core API
is implemented; BitZ/Flock migration and full performance qualification are still
in progress. See [API.md](API.md) for types and signatures,
[PLAN.md](PLAN.md) for the migration plan, and
[IMPLEMENTATION.md](IMPLEMENTATION.md) for current evidence and remaining work.

```rust
use field::*;

let field = create_prime_field(Uint::from_words([17]));
let weights = [field.from_integer(&3u64), field.from_integer(&7u64)];
let integers = [4u64, 5];
let claim = field.reduce(field.batch_mul_acc(&weights, &integers));
assert_eq!(field.to_integer(&claim), Uint::from_words([13]));

// Reuse output and scratch in repeated work.
let mut out = [field.zero(); 2];
let mut scratch = [field.zero(); 2];
field.batch_invert_or_zero_ct_into(&weights, &mut out, &mut scratch);
```

Runtime elements are compact and do not carry a modulus or lifetime brand. The
caller pairs elements and accumulators with their originating runtime context.
Static prime specifications have distinct Rust types. Integer inputs can exceed
the modulus; mixed MAC consumes them directly without converting a whole table.

```rust
field::define_prime_field! {
    pub Seventeen {
        limbs: 1, modulus: [17],
        element: SeventeenElement, context: SeventeenField,
    }
}
```

Local validation:

```sh
cargo test --manifest-path vendor/field/Cargo.toml --offline --all-features
cargo doc --manifest-path vendor/field/Cargo.toml --offline --no-deps
```

The `serde` feature supplies binary-field serialization; `spongefish` retains
canonical GF128 and static-prime transcript codecs. ARM PMULL uses the `aes` target feature. The x86
GF128 kernel requires both `pclmulqdq` and `sse4.1`; other builds use the portable
kernel. `field::gf128::KERNEL` reports the selected GF128 implementation.

`_ct` operations use fixed declared widths and masked validity. Public sampling
has explicitly variable-time interfaces. Timing properties must also be checked
in compiled code on each supported target. `Gf128` and `Gf8` are the shared binary type names; old `F128`, `Fq`, and
`FqDefault` compatibility names are removed. Public window-table exponentiation
is explicitly named `FixedBasePow::pow_public`; private exponents use `pow_ct`.
