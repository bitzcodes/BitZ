//! Segmented storage for integers whose width is declared by the circuit.
//!
//! Each segment owns `Vec<Z<L>>`. Only the storage dispatch erases `L`; arithmetic
//! still receives the original limb count. Values never choose their own width.

use std::any::Any;
use std::fmt::Debug;
use std::ops::Index;

use field::Z;

trait Segment: Any + Debug + Send + Sync {
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn clone_box(&self) -> Box<dyn Segment>;
    fn limbs(&self) -> usize;
    fn words(&self) -> &[u64];
    fn copy_row_to(&self, index: usize, output: &mut IntegerTable);
}

#[derive(Clone, Debug)]
struct TypedSegment<const L: usize>(Vec<Z<L>>);

impl<const L: usize> Segment for TypedSegment<L> {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Segment> {
        Box::new(self.clone())
    }
    fn limbs(&self) -> usize {
        L
    }
    fn words(&self) -> &[u64] {
        // Z<L> and Uint<L> are repr(transparent) wrappers over [u64; L].
        // The allocation contains exactly len * L initialized, aligned words;
        // this immutable view cannot outlive or mutate its owning segment.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast(), self.0.len() * L) }
    }
    fn copy_row_to(&self, index: usize, output: &mut IntegerTable) {
        output.push(self.0[index]);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Location {
    segment: u32,
    index: u32,
}

/// Logical rows backed by separate fixed-width integer segments.
#[derive(Debug, Default)]
pub struct IntegerTable {
    segments: Vec<Box<dyn Segment>>,
    order: Vec<Location>,
}

impl Clone for IntegerTable {
    fn clone(&self) -> Self {
        Self {
            segments: self.segments.iter().map(|s| s.clone_box()).collect(),
            order: self.order.clone(),
        }
    }
}
impl PartialEq for IntegerTable {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}
impl Eq for IntegerTable {}

impl IntegerTable {
    /// Copies a row while preserving its declared type-level width.
    pub fn copy_row_to(&self, index: usize, output: &mut Self) {
        let loc = self.order[index];
        self.segments[loc.segment as usize].copy_row_to(loc.index as usize, output);
    }
    /// Appends an integer without inspecting its value or removing sign limbs.
    pub fn push<const L: usize>(&mut self, value: Z<L>) {
        let segment = self
            .segments
            .iter()
            .position(|s| s.limbs() == L)
            .unwrap_or_else(|| {
                self.segments.push(Box::new(TypedSegment::<L>(Vec::new())));
                self.segments.len() - 1
            });
        let values = &mut self.segments[segment]
            .as_any_mut()
            .downcast_mut::<TypedSegment<L>>()
            .expect("one typed segment per declared width")
            .0;
        let location = Location {
            segment: u32::try_from(segment).expect("too many declared widths"),
            index: u32::try_from(values.len()).expect("too many integer rows"),
        };
        values.push(value);
        self.order.push(location);
    }

    pub fn set<const L: usize>(&mut self, index: usize, value: Z<L>) {
        let location = self.order[index];
        let segment = self.segments[location.segment as usize]
            .as_any_mut()
            .downcast_mut::<TypedSegment<L>>()
            .expect("replacement must retain its declared width");
        segment.0[location.index as usize] = value;
    }
    pub fn len(&self) -> usize {
        self.order.len()
    }
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
    /// Maximum declared width; independent of all stored values.
    pub fn max_limbs(&self) -> usize {
        self.segments.iter().map(|s| s.limbs()).max().unwrap_or(0)
    }
    pub fn payload_bytes(&self) -> usize {
        self.order.len() * std::mem::size_of::<Location>()
            + self
                .segments
                .iter()
                .map(|s| std::mem::size_of_val(s.words()))
                .sum::<usize>()
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[u64]> + DoubleEndedIterator {
        self.order.iter().map(|loc| {
            let segment = &self.segments[loc.segment as usize];
            let width = segment.limbs();
            &segment.words()[loc.index as usize * width..(loc.index as usize + 1) * width]
        })
    }

    /// Resolves segment dispatch once before a row-processing kernel.
    /// This allocates one descriptor per declared width, never one per row.
    pub fn view(&self) -> IntegerTableView<'_> {
        IntegerTableView {
            order: &self.order,
            segments: self
                .segments
                .iter()
                .map(|s| (s.limbs(), s.words()))
                .collect(),
        }
    }
}

impl Index<usize> for IntegerTable {
    type Output = [u64];
    fn index(&self, index: usize) -> &Self::Output {
        let loc = self.order[index];
        let segment = &self.segments[loc.segment as usize];
        let width = segment.limbs();
        &segment.words()[loc.index as usize * width..(loc.index as usize + 1) * width]
    }
}

/// Borrowed row lookup with the storage dispatch already resolved.
pub struct IntegerTableView<'a> {
    order: &'a [Location],
    segments: Vec<(usize, &'a [u64])>,
}
impl IntegerTableView<'_> {
    pub fn len(&self) -> usize {
        self.order.len()
    }
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}
impl Index<usize> for IntegerTableView<'_> {
    type Output = [u64];
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let loc = self.order[index];
        let (width, words) = self.segments[loc.segment as usize];
        &words[loc.index as usize * width..(loc.index as usize + 1) * width]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_widths_preserve_zero_sign_and_layout() {
        let mut table = IntegerTable::default();
        table.push(Z::<1>::ZERO);
        table.push(Z::<9>::ZERO);
        table.push(Z::<1>::from_twos_complement_words([u64::MAX]));
        table.push(Z::<9>::from_twos_complement_words([u64::MAX; 9]));
        table.push(Z::<2>::from_twos_complement_words([0, 1 << 63]));
        assert_eq!(
            table.iter().map(<[u64]>::len).collect::<Vec<_>>(),
            [1, 9, 1, 9, 2]
        );
        assert_eq!(table.max_limbs(), 9);
        assert_eq!(table.payload_bytes(), 5 * 8 + 22 * 8);
        assert_eq!(table, table.clone());
        let view = table.view();
        for (index, words) in table.iter().enumerate() {
            assert_eq!(&view[index], words);
        }
        assert_eq!(view[3], [u64::MAX; 9]);
        assert_eq!(view[4], [0, 1 << 63]);
    }
}

/// Unsigned counterpart of `IntegerTable`, retaining each declared `Uint<L>`
/// segment. The backing table is a word store; no signed arithmetic is used.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UnsignedIntegerTable {
    words: IntegerTable,
}
impl UnsignedIntegerTable {
    pub fn push<const L: usize>(&mut self, value: field::Uint<L>) {
        self.words
            .push(Z::from_twos_complement_words(*value.as_words()));
    }
    pub fn set<const L: usize>(&mut self, index: usize, value: field::Uint<L>) {
        self.words
            .set(index, Z::from_twos_complement_words(*value.as_words()));
    }
    pub fn copy_row_to(&self, index: usize, output: &mut Self) {
        self.words.copy_row_to(index, &mut output.words);
    }
    pub fn len(&self) -> usize {
        self.words.len()
    }
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[u64]> + DoubleEndedIterator {
        self.words.iter()
    }
    pub fn payload_bytes(&self) -> usize {
        self.words.payload_bytes()
    }
    pub fn view(&self) -> IntegerTableView<'_> {
        self.words.view()
    }
}
impl Index<usize> for UnsignedIntegerTable {
    type Output = [u64];
    fn index(&self, index: usize) -> &[u64] {
        &self.words[index]
    }
}
