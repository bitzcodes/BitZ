// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Basis entries from Flock/Binius PHI_8_TABLE; see VENDORED.md.
use crate::Gf128;
pub(super) const PHI8_BASIS: [Gf128; 8] = [
    Gf128::new(0x0000000000000001, 0x0000000000000000),
    Gf128::new(0x6b8330483c2e9849, 0x0dcb364640a222fe),
    Gf128::new(0x7573da4a5f7710ed, 0x3d5bd35c94646a24),
    Gf128::new(0x41a12db1f974f3ac, 0x6d58c4e181f9199f),
    Gf128::new(0x5e2f716f4ede412f, 0xa72ec17764d7ced5),
    Gf128::new(0x5cb10fbabcf00118, 0x4d52354a3a3d8c86),
    Gf128::new(0x95ed1f57f3632d4d, 0x553e92e8bc0ae9a7),
    Gf128::new(0x512625b1f09fa87e, 0x93252331bf042b11),
];

/// AES subfield images, for indexing with public interpolation nodes only.
/// For private bytes use `embed_gf8` or `Gf128Ops::embed`.
pub const AES_EMBEDDING_TABLE: [Gf128; 256] = {
    let mut table = [Gf128::ZERO; 256];
    let mut value = 0;
    while value < 256 {
        let mut i = 0;
        while i < 8 {
            let mask = 0u64.wrapping_sub(((value >> i) & 1) as u64);
            table[value].lo ^= PHI8_BASIS[i].lo & mask;
            table[value].hi ^= PHI8_BASIS[i].hi & mask;
            i += 1;
        }
        value += 1;
    }
    table
};
