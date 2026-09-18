# SHA-256 with secp256k1 key recovery

The `sha256sign` example hashes a variable-length message, recovers
a secp256k1 public key from the digest and signature, and checks an address
derived from bytes 12..32 of SHA-256 of the recovered key. The message, length,
signature, recovery bit, and address are public input/output wires.

This example uses current `ByteVec`, `sha256_varlen`, and `ecrecover` primitives.
SHA-256 emits four big-endian digest words; reversing their order converts the
digest to the little-endian limbs used by `BigUint`. Recovered coordinates are
encoded as big-endian bytes before hashing. Address comparison uses the
little-endian byte packing expected by `ByteVec`.

The witness generator uses a seeded RNG, chooses a length from 1 through the
configured maximum, and signs during witness population. It does not consume
the shared F2Z fixture. Maximum lengths use `u32`; malformed or zero benchmark
lengths fail instead of silently selecting the default.

```sh
# One signature, at most 320 message bytes (up to six message compressions).
# Hashing the recovered 64-byte public key adds two compressions.
cargo bench -p binius-examples --bench sha256sign

N_SIGNATURES=4 MESSAGE_MAX_BYTES=256 cargo bench -p binius-examples --bench sha256sign

# The current CLI registers a named circuit subcommand.
cargo run --release -p binius-examples --example sha256sign -- sha256sign prove -n 1 -m 128
```

The matched SHA-chain/P-256 comparison is the separate
`binius_circuits::sha256_ecdsa::Sha256Ecdsa` relation. It has a fixed private
message and public `(i,Qx,Qy,r,s)`, accepts standard P-256 signatures including
high-s, and adds no public-key hash. The F2Z repository's isolated worker supplies
shared fixtures, explicit non-ZK proof parameters, and separate setup/per-proof
measurements for that relation.
