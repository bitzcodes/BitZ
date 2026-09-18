//! Packed/factored input preparation. No transcript or proving loop lives here.
use super::*;
use crate::sumcheck::inner::input;

pub struct PackedInput<'a, S: ?Sized, H: ?Sized> {
    coefficients: &'a S,
    bits: &'a H,
    num_vars: usize,
    live_len: usize,
    prefix: usize,
}
impl<'a, S: ?Sized, H: ?Sized> PackedInput<'a, S, H> {
    pub(crate) fn new(
        coefficients: &'a S,
        bits: &'a H,
        num_vars: usize,
        live_len: usize,
        prefix: usize,
    ) -> Self {
        Self {
            coefficients,
            bits,
            num_vars,
            live_len,
            prefix,
        }
    }
}
pub struct PrefixState<'a, S: ?Sized, H: ?Sized, const K: usize> {
    input: PackedInput<'a, S, H>,
    round: usize,
    accumulators: Option<PrefixAccumulators>,
    lagrange: Vec<Field>,
    point: [Field; SHA256_INNER_PREFIX_MAX_VARS],
    table: Option<CompactPrefixVTable>,
    prefix_weights: Vec<Field>,
    next: [Field; 2],
    stride: usize,
}
impl<'a, S: InnerSumcheckMleSource + ?Sized, H: Sha256InnerBitSource + ?Sized, const K: usize>
    PrefixState<'a, S, H, K>
{
    fn new(input: PackedInput<'a, S, H>, f: &FieldConfig) -> Result<Self, SumcheckError> {
        let zero = f.zero();
        validate_inputs::<K, _>(input.num_vars, input.live_len, input.bits)?;
        if input
            .coefficients
            .declared_num_vars()
            .is_some_and(|n| n != input.num_vars)
        {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        input.coefficients.validate_shape(input.live_len, f)?;
        let accumulators = if K > 0 {
            Some(input.coefficients.build_prefix_accumulators::<K, _>(
                input.num_vars,
                input.live_len,
                input.bits,
                f,
                &zero,
            )?)
        } else {
            None
        };
        let mut state = Self {
            input,
            round: 0,
            accumulators,
            lagrange: if K > 0 { vec![f.one()] } else { Vec::new() },
            point: [zero; SHA256_INNER_PREFIX_MAX_VARS],
            table: None,
            prefix_weights: Vec::new(),
            next: [zero; 2],
            stride: 1,
        };
        if K == 0 {
            state.prepare_tail(f)?;
        }
        Ok(state)
    }
    fn prepare_tail(&mut self, f: &FieldConfig) -> Result<(), SumcheckError> {
        self.accumulators = None;
        self.lagrange = Vec::new();
        let (zero, one) = (f.zero(), f.one());
        let table = fold_prefix_v_table::<K, _>(
            self.input.num_vars,
            self.input.live_len,
            self.input.coefficients,
            &self.point[..K],
            f,
            &zero,
            &one,
        )?;
        self.prefix_weights = equality_weights_lsb(&self.point[..K], &zero, &one, f);
        if self.input.num_vars > K {
            self.next = sum_first_tail_round::<K, _>(
                &table,
                self.input.live_len,
                self.input.bits,
                &self.prefix_weights,
                f,
                &zero,
                &one,
            )?;
        }
        self.table = Some(table);
        Ok(())
    }
    fn coefficients(&self, f: &FieldConfig) -> Result<[Field; 2], SumcheckError> {
        if self.round < K {
            let [c2, c0] = self.accumulators.as_ref().unwrap().evaluate_round(
                self.round,
                &self.lagrange,
                f,
            )?;
            Ok([c0, c2])
        } else {
            Ok(self.next)
        }
    }
    fn fold(&mut self, f: &FieldConfig, r: &Field) -> Result<(), SumcheckError> {
        let (zero, one) = (f.zero(), f.one());
        if self.round < K {
            self.point[self.round] = *r;
            extend_lagrange_coefficients(&mut self.lagrange, r, &one, &zero, f);
            self.round += 1;
            if self.round == K {
                self.prepare_tail(f)?;
            }
            return Ok(());
        }
        let table = self.table.as_mut().unwrap();
        let more = self.round + 1 < self.input.num_vars;
        if self.round == K {
            if more {
                self.next = fold_first_tail_round_and_prepare_next_in_place::<K, _>(
                    table,
                    self.input.live_len,
                    self.input.bits,
                    &self.prefix_weights,
                    r,
                    f,
                    &zero,
                    &one,
                    f,
                )?;
            } else {
                fold_first_tail_round_in_place::<K, _>(
                    table,
                    self.input.live_len,
                    self.input.bits,
                    &self.prefix_weights,
                    r,
                    f,
                    &zero,
                    &one,
                )?;
            }
        } else {
            if more {
                self.next = fold_interleaved_and_prepare_next_round_in_place(
                    &mut table.values,
                    self.stride,
                    r,
                    f,
                    &zero,
                )?;
            } else {
                fold_interleaved_in_place(&mut table.values, self.stride, r, f, &zero);
            }
            self.stride *= 2;
        }
        self.round += 1;
        Ok(())
    }
    fn terminal(&self, f: &FieldConfig) -> Result<[Field; 2], SumcheckError> {
        let table = self.table.as_ref().unwrap();
        let v = field_from_raw(&table.values[0], f);
        let h = if self.input.num_vars == K {
            folded_packed_h::<K, _>(
                0,
                self.input.live_len,
                self.input.bits,
                &self.prefix_weights,
                f,
                &f.zero(),
                &f.one(),
            )?
        } else {
            field_from_raw(&table.values[1], f)
        };
        Ok([v, h])
    }
}

pub enum State<'a, S: ?Sized, H: ?Sized> {
    K0(PrefixState<'a, S, H, 0>),
    K1(PrefixState<'a, S, H, 1>),
    K2(PrefixState<'a, S, H, 2>),
    K3(PrefixState<'a, S, H, 3>),
    K4(PrefixState<'a, S, H, 4>),
}
macro_rules! dispatch {
    ($state:expr,$s:ident,$body:expr) => {
        match $state {
            State::K0($s) => $body,
            State::K1($s) => $body,
            State::K2($s) => $body,
            State::K3($s) => $body,
            State::K4($s) => $body,
        }
    };
}
impl<S: ?Sized, H: ?Sized> input::sealed::Input for PackedInput<'_, S, H> {}
impl<'a, S: InnerSumcheckMleSource + ?Sized, H: Sha256InnerBitSource + ?Sized>
    input::Input<FieldConfig> for PackedInput<'a, S, H>
{
    type Weights = ();
    type State = State<'a, S, H>;
    type Codec = input::Canonical;
    fn prepare(self, f: &FieldConfig, _: ()) -> Result<Self::State, SumcheckError> {
        Ok(match self.prefix {
            0 => State::K0(PrefixState::new(self, f)?),
            1 => State::K1(PrefixState::new(self, f)?),
            2 => State::K2(PrefixState::new(self, f)?),
            3 => State::K3(PrefixState::new(self, f)?),
            4 => State::K4(PrefixState::new(self, f)?),
            _ => return Err(SumcheckError::InvalidProductDimensions),
        })
    }
}
impl<S: InnerSumcheckMleSource + ?Sized, H: Sha256InnerBitSource + ?Sized> input::State<FieldConfig>
    for State<'_, S, H>
{
    fn num_vars(&self) -> usize {
        dispatch!(self, s, s.input.num_vars)
    }
    fn validate_claim(&self, f: &FieldConfig, claim: &Field) -> Result<(), SumcheckError> {
        validate_field_value(claim, f)
    }
    fn coefficients(&self, f: &FieldConfig) -> Result<[Field; 2], SumcheckError> {
        dispatch!(self, s, s.coefficients(f))
    }
    fn fold(&mut self, f: &FieldConfig, r: &Field) -> Result<(), SumcheckError> {
        dispatch!(self, s, s.fold(f, r))
    }
    fn terminal(&self, f: &FieldConfig) -> Result<[Field; 2], SumcheckError> {
        dispatch!(self, s, s.terminal(f))
    }
}
