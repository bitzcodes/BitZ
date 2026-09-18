//! Versioned inputs and logical output identity shared with Limber's int_mult.
pub const INPUT_DOMAIN: &[u8] = b"native-mul/mod32/inputs/v1";
pub const ROW_DOMAIN: &[u8] = b"native-mul/mod32/rows/v1";

pub fn inputs(exponent: usize, seed: u64) -> Vec<(u64, u64)> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(INPUT_DOMAIN);
    hasher.update(&seed.to_le_bytes());
    hasher.update(
        &u32::try_from(exponent)
            .expect("exponent fits u32")
            .to_le_bytes(),
    );
    let mut stream = hasher.finalize_xof();
    (0..1usize << exponent)
        .map(|_| {
            let mut bytes = [0u8; 8];
            stream.fill(&mut bytes);
            (
                u64::from(u32::from_le_bytes(bytes[..4].try_into().unwrap())),
                u64::from(u32::from_le_bytes(bytes[4..].try_into().unwrap())),
            )
        })
        .collect()
}

pub fn digest_rows(rows: impl IntoIterator<Item = [u64; 3]>, count: usize) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(ROW_DOMAIN);
    hasher.update(&(count as u64).to_le_bytes());
    let mut actual = 0;
    for row in rows {
        for value in row {
            hasher.update(
                &u32::try_from(value)
                    .expect("canonical mod32 limb")
                    .to_le_bytes(),
            );
        }
        actual += 1;
    }
    assert_eq!(actual, count, "canonical row count");
    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cross_repository_golden_vectors() {
        let seed = 0x5533_3250_4353_0064;
        for (exponent, expected) in [
            (
                4,
                "90f3ca71e3e95a08eb8960a5daea013132ea10b2f69784b2b96604683d417f0d",
            ),
            (
                15,
                "a006d0e2143cfce1ec9dc60dd92be801f48a126b8076d4704cb698e7fd4ac9da",
            ),
        ] {
            let pairs = inputs(exponent, seed);
            if exponent == 4 {
                assert_eq!(
                    &pairs[..4],
                    &[
                        (1210304475, 2365989708),
                        (1744110415, 2023825938),
                        (587259919, 3206144740),
                        (686439776, 1601866294)
                    ]
                );
            }
            assert_eq!(
                digest_rows(
                    pairs.iter().map(|&(x, y)| [x, y, (x * y) & 0xffff_ffff]),
                    pairs.len()
                ),
                expected
            );
        }
    }
}
