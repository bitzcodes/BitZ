//! Coefficient storage is independent of graph topology and execution fields.
use std::{borrow::Cow, sync::Arc};

use crate::integer_storage::IntegerTable;
use field::{IntegerEmbedding, RingOps, Z, ZRef};

pub trait CoefficientStore {
    type Ref<'a>
    where
        Self: 'a;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, index: usize) -> Self::Ref<'_>;
    /// Preparation visits storage slots once, even when many entries share a coefficient.
    fn distinct_len(&self) -> usize {
        self.len()
    }
    fn distinct_coefficient(&self, index: usize) -> Self::Ref<'_> {
        self.get(index)
    }
    fn coefficient_index(&self, entry: usize) -> usize {
        entry
    }
    /// Optionally discard unused slots, returning old-to-new indices. Storage
    /// owns this policy; compilation never compares or hashes coefficients.
    fn compact(&mut self, _used: &[bool]) -> Option<Vec<usize>> {
        None
    }
}

pub trait StoreCoefficient<C>: CoefficientStore {
    fn store(&mut self, coefficient: C) -> usize;
}

impl<C> CoefficientStore for Box<[C]> {
    type Ref<'a>
        = &'a C
    where
        C: 'a;
    fn len(&self) -> usize {
        self.as_ref().len()
    }
    fn get(&self, index: usize) -> &C {
        &self[index]
    }
}
impl<S: CoefficientStore + ?Sized> CoefficientStore for Arc<S> {
    type Ref<'a>
        = S::Ref<'a>
    where
        Self: 'a;
    fn len(&self) -> usize {
        (**self).len()
    }
    fn get(&self, index: usize) -> Self::Ref<'_> {
        (**self).get(index)
    }
    fn distinct_len(&self) -> usize {
        (**self).distinct_len()
    }
    fn distinct_coefficient(&self, index: usize) -> Self::Ref<'_> {
        (**self).distinct_coefficient(index)
    }
    fn coefficient_index(&self, entry: usize) -> usize {
        (**self).coefficient_index(entry)
    }
}
impl<S: CoefficientStore + ?Sized> CoefficientStore for &S {
    type Ref<'a>
        = S::Ref<'a>
    where
        Self: 'a;
    fn len(&self) -> usize {
        (**self).len()
    }
    fn get(&self, index: usize) -> Self::Ref<'_> {
        (**self).get(index)
    }
    fn distinct_len(&self) -> usize {
        (**self).distinct_len()
    }
    fn distinct_coefficient(&self, index: usize) -> Self::Ref<'_> {
        (**self).distinct_coefficient(index)
    }
    fn coefficient_index(&self, entry: usize) -> usize {
        (**self).coefficient_index(entry)
    }
}

impl<C> CoefficientStore for Vec<C> {
    type Ref<'a>
        = &'a C
    where
        C: 'a;
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn get(&self, index: usize) -> &C {
        &self[index]
    }
    fn compact(&mut self, used: &[bool]) -> Option<Vec<usize>> {
        let mut remap = vec![usize::MAX; self.len()];
        let mut i = 0;
        let mut n = 0;
        self.retain(|_| {
            let keep = used[i];
            if keep {
                remap[i] = n;
                n += 1;
            }
            i += 1;
            keep
        });
        Some(remap)
    }
}
impl<C> StoreCoefficient<C> for Vec<C> {
    fn store(&mut self, value: C) -> usize {
        let index = self.len();
        self.push(value);
        index
    }
}
impl CoefficientStore for IntegerTable {
    type Ref<'a> = ZRef<'a>;
    fn len(&self) -> usize {
        self.len()
    }
    fn get(&self, index: usize) -> ZRef<'_> {
        ZRef::from_twos_complement_words(&self[index])
    }
    fn compact(&mut self, used: &[bool]) -> Option<Vec<usize>> {
        let mut out = Self::default();
        let mut remap = vec![usize::MAX; self.len()];
        for (i, &live) in used.iter().enumerate() {
            if live {
                remap[i] = out.len();
                self.copy_row_to(i, &mut out);
            }
        }
        *self = out;
        Some(remap)
    }
}
impl<const L: usize> StoreCoefficient<Z<L>> for IntegerTable {
    fn store(&mut self, value: Z<L>) -> usize {
        let index = self.len();
        self.push(value);
        index
    }
}

/// Elements must originate from `field`, including its runtime modulus.
/// Keeping the provider here prevents preparation with a second context.
pub struct FieldCoefficients<'f, F: RingOps> {
    pub(super) field: &'f F,
    pub(super) values: Vec<F::Elem>,
}
impl<'f, F: RingOps> FieldCoefficients<'f, F> {
    pub fn new(field: &'f F) -> Self {
        Self {
            field,
            values: Vec::new(),
        }
    }
    pub fn field(&self) -> &'f F {
        self.field
    }
}
impl<F: RingOps> CoefficientStore for FieldCoefficients<'_, F> {
    type Ref<'a>
        = &'a F::Elem
    where
        Self: 'a;
    fn len(&self) -> usize {
        self.values.len()
    }
    fn get(&self, index: usize) -> &F::Elem {
        &self.values[index]
    }
    fn compact(&mut self, used: &[bool]) -> Option<Vec<usize>> {
        CoefficientStore::compact(&mut self.values, used)
    }
}
impl<F: RingOps> StoreCoefficient<F::Elem> for FieldCoefficients<'_, F> {
    fn store(&mut self, value: F::Elem) -> usize {
        let index = self.values.len();
        self.values.push(value);
        index
    }
}

impl<C> super::WengertTape<Vec<C>> {
    pub fn prepare<'a, F>(&'a self, field: &'a F) -> super::PreparedWengert<'a, &'a F>
    where
        F: RingOps + Sync + IntegerEmbedding<C>,
    {
        let values = self
            .coefficients
            .iter()
            .map(|c| field.from_integer(c))
            .collect();
        super::PreparedWengert::new(&self.graph, field, Cow::Owned(values))
    }
}
impl super::WengertTape<IntegerTable> {
    pub fn prepare<'a, F>(&'a self, field: &'a F) -> super::PreparedWengert<'a, &'a F>
    where
        F: RingOps + Sync,
        for<'c> F: IntegerEmbedding<ZRef<'c>>,
    {
        self.prepare_projected(field, |words| {
            field.from_integer(&ZRef::from_twos_complement_words(words))
        })
    }

    /// Prepare a reusable wide-integer projection once, then project each
    /// distinct coefficient. Execution never performs integer lookup.
    pub fn prepare_projected<'a, F>(
        &'a self,
        field: &'a F,
        project: impl Fn(&[u64]) -> F::Elem,
    ) -> super::PreparedWengert<'a, &'a F>
    where
        F: RingOps + Sync,
    {
        let view = self.coefficients.view();
        let values = (0..view.len()).map(|i| project(&view[i])).collect();
        super::PreparedWengert::new(&self.graph, field, Cow::Owned(values))
    }
}
impl<F: RingOps + Sync> super::WengertTape<FieldCoefficients<'_, F>> {
    pub fn prepare(&self) -> super::PreparedWengert<'_, &F> {
        super::PreparedWengert::new(
            &self.graph,
            self.coefficients.field,
            Cow::Borrowed(&self.coefficients.values),
        )
    }
}
