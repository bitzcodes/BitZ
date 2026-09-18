use crate::field::Gf128;

#[inline]
pub(super) fn fold_pairs(src: &[Gf128], base: usize, dst: &mut [Gf128], r: Gf128) {
    let one_plus_r = Gf128::ONE + r;
    for (t, value) in dst.iter_mut().enumerate() {
        let s = 2 * (base + t);
        *value = src[s] * one_plus_r + src[s + 1] * r;
    }
}
