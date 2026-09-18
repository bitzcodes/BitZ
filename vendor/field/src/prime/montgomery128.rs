//! Scalar schedule retained from BitZ's raw Montgomery context.
#[cfg(not(target_arch = "aarch64"))]
use crate::{CtMask, CtSelect};

/// Keep each pair of limbs together for load/store scheduling. PrimeValue has
/// word alignment, so these accesses explicitly allow unaligned u128 storage.
#[cfg(target_endian = "little")]
#[inline]
pub(super) fn batch_mul<Id, const L: usize>(
    params: &super::PrimeParameters<L>,
    lhs: &[super::PrimeValue<Id, L>],
    rhs: &[super::PrimeValue<Id, L>],
    out: &mut [super::PrimeValue<Id, L>],
) {
    assert_eq!(L, 2);
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), out.len());
    let modulus = [params.modulus.0[0], params.modulus.0[1]];
    for i in 0..lhs.len() {
        // SAFETY: L==2 and repr(transparent) guarantee 16 initialized bytes per
        // value. The public shape checks cover i in every slice. Little-endian
        // targets map the first limb to the low u128 half. read/write_unaligned
        // require no extra alignment; the unique output borrow forbids aliasing
        // with either input. Every output is a canonical Montgomery encoding.
        unsafe {
            let a = lhs.as_ptr().cast::<u128>().add(i).read_unaligned();
            let b = rhs.as_ptr().cast::<u128>().add(i).read_unaligned();
            let result = mul(
                [a as u64, (a >> 64) as u64],
                [b as u64, (b >> 64) as u64],
                modulus,
                params.neg_inv,
            );
            out.as_mut_ptr()
                .cast::<u128>()
                .add(i)
                .write_unaligned(result[0] as u128 | ((result[1] as u128) << 64));
        }
    }
}

#[inline(always)]
pub(super) fn mul(a: [u64; 2], b: [u64; 2], modulus: [u64; 2], neg_inv: u64) -> [u64; 2] {
    let p00 = a[0] as u128 * b[0] as u128;
    let p01 = a[0] as u128 * b[1] as u128;
    let p10 = a[1] as u128 * b[0] as u128;
    let p11 = a[1] as u128 * b[1] as u128;
    let mid = (p00 >> 64) + (p01 as u64) as u128 + (p10 as u64) as u128;
    let mid2 = (mid >> 64) + (p01 >> 64) + (p10 >> 64) + (p11 as u64) as u128;
    redc(
        [
            p00 as u64,
            mid as u64,
            mid2 as u64,
            ((mid2 >> 64) + (p11 >> 64)) as u64,
        ],
        modulus,
        neg_inv,
    )
}

/// Canonical REDC under the caller-established `input < modulus * 2^128` bound.
#[inline(always)]
pub(super) fn redc(value: [u64; 4], modulus: [u64; 2], neg_inv: u64) -> [u64; 2] {
    let [q0, q1] = modulus;
    let [t0, t1, t2, t3] = value;
    let m0 = t0.wrapping_mul(neg_inv);
    let mq0 = m0 as u128 * q0 as u128;
    let mq1 = m0 as u128 * q1 as u128;
    let c0 = t0 as u128 + (mq0 as u64) as u128;
    debug_assert_eq!(c0 as u64, 0);
    let c1 = t1 as u128 + (mq0 >> 64) + (mq1 as u64) as u128 + (c0 >> 64);
    let c2 = t2 as u128 + (mq1 >> 64) + (c1 >> 64);
    let c3 = t3 as u128 + (c2 >> 64);
    let m1 = (c1 as u64).wrapping_mul(neg_inv);
    let n0 = m1 as u128 * q0 as u128;
    let n1 = m1 as u128 * q1 as u128;
    let d1 = (c1 as u64) as u128 + (n0 as u64) as u128;
    debug_assert_eq!(d1 as u64, 0);
    let d2 = (c2 as u64) as u128 + (n0 >> 64) + (n1 as u64) as u128 + (d1 >> 64);
    let d3 = (c3 as u64) as u128 + (n1 >> 64) + (d2 >> 64);
    debug_assert!((c3 >> 64) + (d3 >> 64) <= 1);
    // The canonical-input bound makes these carries mutually exclusive.
    let overflow = (c3 >> 64) as u64 | (d3 >> 64) as u64;
    conditional_subtract([d2 as u64, d3 as u64], modulus, overflow)
}

#[inline(always)]
fn conditional_subtract(value: [u64; 2], modulus: [u64; 2], overflow: u64) -> [u64; 2] {
    #[cfg(target_arch = "aarch64")]
    {
        let [mut lo, mut hi] = value;
        // SAFETY: fixed register-only compare/select/subtract. This mirrors
        // the branch-free sequence of the production raw context, including
        // the 129th carry bit; neither operands nor carry affect control flow.
        unsafe {
            core::arch::asm!(
                "cmp {lo}, {q0}", "sbcs xzr, {hi}, {q1}",
                "ccmp {overflow}, #0, #0, lo",
                "csel {sub_hi}, {q1}, xzr, ne", "csel {sub_lo}, {q0}, xzr, ne",
                "subs {lo}, {lo}, {sub_lo}", "sbc {hi}, {hi}, {sub_hi}",
                lo=inout(reg) lo,hi=inout(reg) hi,q0=in(reg) modulus[0],q1=in(reg) modulus[1],
                overflow=in(reg) overflow,sub_lo=out(reg) _,sub_hi=out(reg) _,
                options(pure,nomem,nostack),
            );
        }
        [lo, hi]
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let [q0, q1] = modulus;
        let [r0, r1] = value;
        let result = r0 as u128 | (r1 as u128) << 64;
        let p = q0 as u128 | (q1 as u128) << 64;
        let (reduced, borrow) = result.overflowing_sub(p);
        let mask = CtMask::from_lsb(overflow | (!borrow as u64));
        [
            u64::ct_select(&(result as u64), &(reduced as u64), mask),
            u64::ct_select(&((result >> 64) as u64), &((reduced >> 64) as u64), mask),
        ]
    }
}

// Retained from circuit Wengert at 6271724d; final selection uses masks.
/// Two-limb FIOS Montgomery multiplication with a final canonical reduction.
#[inline(always)]
pub(super) fn mul_fios(
    left: [u64; 2],
    right: [u64; 2],
    modulus: [u64; 2],
    mod_neg_inv: u64,
) -> [u64; 2] {
    let mut output = [0_u64; 2];
    let mut meta_carry = 0_u128;
    for left_limb in left {
        let low_product = u128::from(left_limb) * u128::from(right[0]) + u128::from(output[0]);
        let multiplier = (low_product as u64).wrapping_mul(mod_neg_inv);
        let (sum, overflow) =
            (u128::from(multiplier) * u128::from(modulus[0])).overflowing_add(low_product);
        let mut carry = (u128::from(overflow) << 64) | (sum >> 64);

        let high_product = u128::from(left_limb) * u128::from(right[1]) + u128::from(output[1]);
        let modulus_product = u128::from(multiplier) * u128::from(modulus[1]) + carry;
        let (sum, overflow) = high_product.overflowing_add(modulus_product);
        output[0] = sum as u64;
        carry = (u128::from(overflow) << 64) | (sum >> 64);

        carry += meta_carry;
        output[1] = carry as u64;
        meta_carry = carry >> 64;
    }

    let (low, low_borrow) = output[0].overflowing_sub(modulus[0]);
    let (high, first_borrow) = output[1].overflowing_sub(modulus[1]);
    let (high, second_borrow) = high.overflowing_sub(u64::from(low_borrow));
    let take =
        crate::CtMask::from_lsb((meta_carry as u64) | u64::from(!(first_borrow | second_borrow)));
    use crate::CtSelect;
    [
        u64::ct_select(&output[0], &low, take),
        u64::ct_select(&output[1], &high, take),
    ]
}
