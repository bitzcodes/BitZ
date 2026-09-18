# /// script
# requires-python = ">=3.10"
# dependencies = ["blake3==1.0.5"]
# ///
"""Independent KAT generator for the limber Poseidon2 benchmark workload.

Reimplemented from the specification in plan/poseidon_modp_bench.md (round
constants, matrices, permutation, chain, benchmark messages) — NOT ported
from the Rust implementation, and it never reads Rust output. The committed
fixture `tests/data/poseidon2_kat_v1.json` is byte-compared against this
generator's canonical JSON.

Modes (mutually exclusive):
  --write <path>   deliberate maintainer update; atomically replace the file
  --check <path>   read-only CI check; regenerate in memory and byte-compare

Both modes verify that the PEP 723 dependency pin above exactly matches the
single line in scripts/requirements-kat.txt; pin drift is an error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

from blake3 import blake3

T = 3
ALPHA = 5
R_F = 8
R_P = 56
ROUNDS = R_F + R_P

RC_DOMAIN = b"limber-poseidon2-v1/rc"
MSG_DOMAIN = b"limber-poseidon2-v1/msg"
SCHEMA_DOMAIN = "limber-poseidon2-v1"
SCHEMA_VERSION = 1

# External and internal linear-layer matrices (the official Poseidon2 t=3
# pair: M_E = circ(2,1,1), M_I = J + diag(1,1,2); deliberately distinct).
M_E = [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
M_I = [[2, 1, 1], [1, 2, 1], [1, 1, 3]]

FIELDS = {
    "bn254": int(
        "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001", 16
    ),
    "bls12_381": int(
        "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", 16
    ),
    "secp256k1": int(
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141", 16
    ),
}

# Fixed oracle for the specialized Algorithms 2-3 determinants
# D_r = det[e0, M^r e0, M^(2r) e0], r = 1..12, of the selected M_I.
D_ORACLE = [
    1,
    84,
    3683,
    133056,
    4453429,
    143857980,
    4562087447,
    143190609408,
    4467253670377,
    138861853051236,
    4306780344808523,
    133388643001078080,
]

IV = (1 << 64) + 0x9E3779B97F4A7C15


def hex32(v: int) -> str:
    """32-byte big-endian lowercase hex, no 0x prefix."""
    return format(v, "064x")


def is_full_round(r: int) -> bool:
    """Whether 1-based round r is a full (external) round."""
    return r <= R_F // 2 or r > R_F // 2 + R_P


# ---------------------------------------------------------------------------
# Matrix security predicates (independent implementation of the same checks
# the Rust build_params runs; neither consumes the other's result).


def mat_vec(m, v, p):
    return [sum(m[i][j] * v[j] for j in range(3)) % p for i in range(3)]


def mat_mul(a, b, p=None):
    out = [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    if p is not None:
        out = [[x % p for x in row] for row in out]
    return out


def is_scalar_matrix(m, p):
    for i in range(3):
        for j in range(3):
            if i != j and m[i][j] % p != 0:
                return False
    return m[0][0] % p == m[1][1] % p == m[2][2] % p


def proportional(a, b, p):
    """All 2x2 minors of the 3x2 matrix [a b] vanish mod p."""
    for i in range(3):
        for j in range(i + 1, 3):
            if (a[i] * b[j] - a[j] * b[i]) % p != 0:
                return False
    return True


def validate_internal_matrix(m_int, p) -> None:
    """The t=3, s=1 subspace-trail checks (Algorithms 1-3 specialization)."""
    m = [[x % p for x in row] for row in m_int]
    m2 = mat_mul(m, m, p)
    if is_scalar_matrix(m, p) or is_scalar_matrix(m2, p):
        raise SystemExit("matrix validation: M or M^2 is scalar")
    # S1 = span(e1, e2) must not be M-invariant.
    if m[0][1] % p == 0 and m[0][2] % p == 0:
        raise SystemExit("matrix validation: span(e1,e2) is M-invariant")
    # S2 = span((0, 1, -1)) must differ from M S2 and M^2 S2.
    v = [0, 1, p - 1]
    if proportional(mat_vec(m, v, p), v, p):
        raise SystemExit("matrix validation: span((0,1,-1)) is M-invariant")
    if proportional(mat_vec(m2, v, p), v, p):
        raise SystemExit("matrix validation: span((0,1,-1)) is M^2-invariant")
    # No eigenline of M inside S1: the unique S1 line with M v back in S1 is
    # (0, a, b) = (0, M[0][2], -M[0][1]); reject if M v is proportional.
    a, b = m[0][2] % p, (-m[0][1]) % p
    u = mat_vec(m, [0, a, b], p)
    assert u[0] == 0
    if (a * u[2] - b * u[1]) % p == 0:
        raise SystemExit("matrix validation: eigenline inside span(e1,e2)")
    # Algorithms 2-3: D_r = det[e0, M^r e0, M^(2r) e0] != 0 for r = 1..12,
    # checked mod p AND against the fixed integer oracle over Z.
    m_r = [row[:] for row in m_int]
    for r in range(1, 13):
        m_2r = mat_mul(m_r, m_r)
        d = m_r[1][0] * m_2r[2][0] - m_r[2][0] * m_2r[1][0]
        if d != D_ORACLE[r - 1]:
            raise SystemExit(f"matrix validation: D_{r} != oracle value")
        if d % p == 0:
            raise SystemExit(f"matrix validation: D_{r} vanishes mod p")
        m_r = mat_mul(m_r, m_int)


def validate_j_plus_diag_mds(m, p) -> None:
    """MDS conditions for M = J + diag(mu - 1), over Z and mod p."""
    for i in range(3):
        for j in range(3):
            if i != j and m[i][j] != 1:
                raise SystemExit("MDS validation: matrix is not J + diag")
    mu = [m[i][i] for i in range(3)]
    for x in mu:
        if x == 0 or x % p == 0 or x == 1 or x % p == 1:
            raise SystemExit("MDS validation: mu in {0, 1}")
    for i in range(3):
        for j in range(i + 1, 3):
            if mu[i] * mu[j] == 1 or (mu[i] * mu[j]) % p == 1:
                raise SystemExit("MDS validation: mu_i * mu_j = 1")
    lhs = mu[0] * mu[1] * mu[2] + 2
    rhs = mu[0] + mu[1] + mu[2]
    if lhs == rhs or lhs % p == rhs % p:
        raise SystemExit("MDS validation: determinant condition violated")


# ---------------------------------------------------------------------------
# Round constants, messages, permutation


def derive_round_constants(p: int) -> list[int]:
    """80 constants in draw order via BLAKE3 XOF rejection sampling."""
    p_be32 = p.to_bytes(32, "big")
    counter = 0
    out = []

    def draw() -> int:
        nonlocal counter
        while True:
            data = (
                RC_DOMAIN
                + p_be32
                + bytes([T, R_F, R_P, ALPHA])
                + counter.to_bytes(4, "big")
            )
            v = int.from_bytes(blake3(data).digest(length=32), "big")
            counter += 1
            if v < p:
                return v

    for r in range(1, ROUNDS + 1):
        lanes = T if is_full_round(r) else 1
        for _ in range(lanes):
            c = draw()
            if c == 0:
                raise SystemExit(f"zero round constant drawn for round {r}")
            out.append(c)
    assert len(out) == 80
    return out


def build_messages(hashes: int) -> list[int]:
    mask = (1 << 250) - 1
    out = []
    for j in range(1, hashes + 1):
        data = MSG_DOMAIN + j.to_bytes(4, "big")
        out.append(int.from_bytes(blake3(data).digest(length=32), "big") & mask)
    return out


def group_constants(flat: list[int]) -> list[list[int]]:
    """Group the 80 draw-order constants into 64 per-round lists."""
    rc, idx = [], 0
    for r in range(1, ROUNDS + 1):
        lanes = T if is_full_round(r) else 1
        rc.append(flat[idx : idx + lanes])
        idx += lanes
    return rc


def permute(p: int, rc: list[list[int]], state: list[int]) -> list[int]:
    assert all(0 <= x < p for x in state)
    s = mat_vec(M_E, state, p)
    for r in range(1, ROUNDS + 1):
        c = rc[r - 1]
        if is_full_round(r):
            s = [pow((s[i] + c[i]) % p, ALPHA, p) for i in range(3)]
            s = mat_vec(M_E, s, p)
        else:
            s[0] = pow((s[0] + c[0]) % p, ALPHA, p)
            s = mat_vec(M_I, s, p)
    return s


def chain(p: int, rc: list[list[int]], messages: list[int]) -> list[int]:
    h, out = IV, []
    for m in messages:
        assert 0 <= m < p
        h = permute(p, rc, [h, m, 0])[0]
        out.append(h)
    return out


# ---------------------------------------------------------------------------
# Fixture assembly and modes


def build_fixture() -> dict:
    messages = build_messages(10)
    fields = {}
    for name, p in FIELDS.items():
        validate_j_plus_diag_mds(M_E, p)
        validate_j_plus_diag_mds(M_I, p)
        validate_internal_matrix(M_I, p)
        flat_rc = derive_round_constants(p)
        rc = group_constants(flat_rc)
        standalone_in = [1, 2, 3]
        standalone_out = permute(p, rc, standalone_in)
        fields[name] = {
            "modulus": hex32(p),
            "m_e": M_E,
            "m_i": M_I,
            "round_constants": [hex32(c) for c in flat_rc],
            "iv": hex32(IV),
            "messages": [hex32(m) for m in messages],
            "standalone_input": [hex32(x) for x in standalone_in],
            "standalone_output": [hex32(x) for x in standalone_out],
            "chain": [hex32(h) for h in chain(p, rc, messages)],
        }
    return {
        "domain": SCHEMA_DOMAIN,
        "schema_version": SCHEMA_VERSION,
        "t": T,
        "alpha": ALPHA,
        "r_f": R_F,
        "r_p": R_P,
        # The circuit's semantic block/public-IO order. A sorted JSON
        # object key order is NOT the semantic order; consumers must use
        # this explicit array.
        "field_order": list(FIELDS.keys()),
        "fields": fields,
    }


def canonical_json(obj: dict) -> bytes:
    """Sorted keys, two-space indentation, one trailing newline."""
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")


def check_pin_consistency() -> None:
    """The PEP 723 pin must exactly match scripts/requirements-kat.txt."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        header = f.read(2048)
    m = re.search(r'dependencies = \["(blake3==[0-9.]+)"\]', header)
    if not m:
        raise SystemExit("pin check: PEP 723 dependency block not found")
    pep723_pin = m.group(1)
    req_path = os.path.join(script_dir, "requirements-kat.txt")
    with open(req_path, encoding="utf-8") as f:
        req_lines = [ln.strip() for ln in f if ln.strip()]
    if req_lines != [pep723_pin]:
        raise SystemExit(
            f"pin check: requirements-kat.txt {req_lines} != PEP 723 pin "
            f"['{pep723_pin}']"
        )


def diff_json(expected: dict, actual: dict, path: str = "$") -> list[str]:
    """Concise path/field diff between two JSON values."""
    diffs: list[str] = []
    if type(expected) is not type(actual):
        return [f"{path}: type {type(actual).__name__} != {type(expected).__name__}"]
    if isinstance(expected, dict):
        for k in sorted(set(expected) | set(actual)):
            if k not in actual:
                diffs.append(f"{path}.{k}: missing")
            elif k not in expected:
                diffs.append(f"{path}.{k}: unexpected")
            else:
                diffs.extend(diff_json(expected[k], actual[k], f"{path}.{k}"))
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: length {len(actual)} != {len(expected)}")
        for i, (e, a) in enumerate(zip(expected, actual)):
            diffs.extend(diff_json(e, a, f"{path}[{i}]"))
    elif expected != actual:
        diffs.append(f"{path}: {actual!r} != {expected!r}")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", metavar="PATH", help="atomically write the fixture")
    mode.add_argument("--check", metavar="PATH", help="read-only drift check")
    args = parser.parse_args()

    check_pin_consistency()
    fixture_bytes = canonical_json(build_fixture())

    if args.write:
        target = os.path.abspath(args.write)
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(target), prefix=".poseidon2_kat_", suffix=".json"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(fixture_bytes)
            os.replace(tmp, target)
        except BaseException:
            os.unlink(tmp)
            raise
        print(f"wrote {len(fixture_bytes)} bytes to {target}")
        return 0

    with open(args.check, "rb") as f:
        on_disk = f.read()
    if on_disk == fixture_bytes:
        print(f"{args.check}: up to date")
        return 0
    print(f"{args.check}: DRIFT from the generator output", file=sys.stderr)
    try:
        diffs = diff_json(json.loads(fixture_bytes), json.loads(on_disk))
        for d in diffs[:50]:
            print(f"  {d}", file=sys.stderr)
        if len(diffs) > 50:
            print(f"  ... and {len(diffs) - 50} more", file=sys.stderr)
        if not diffs:
            print("  (byte-level difference only: whitespace/encoding)", file=sys.stderr)
    except json.JSONDecodeError as e:
        print(f"  fixture is not valid JSON: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
