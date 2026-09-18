use std::array;

use field::ModRingCtx;

use field::Gf128;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Signed, Zero};

use crate::constraints::BoolLinearCombination;
use crate::ecdsa_sha256::{
    VERIFY_2KB_INPUT_BITS, VERIFY_2KB_INTEGER_WITNESS_BITS, VERIFY_2KB_WITNESS_BITS,
    verify_2kb_message_circuit,
};
use crate::matrix_products::{IntegerProducts, MatrixProducts};
use crate::matrix_transpose::MTransposeGenerator;
use crate::p256::{
    VERIFY_DIGEST_INPUT_BITS, VERIFY_DIGEST_INTEGER_WITNESS_BITS, VERIFY_DIGEST_WITNESS_BITS,
    prepare, verify_digest_circuit,
};
use crate::sha256::SHA256_2KB_MESSAGE_BITS;
use crate::stats::Dummy;
use crate::witgen::{PackedWitness, ProductWitgen};
use crate::{Circuit, HintResult, PackedBits, ScalarBits, WitnessContext};

fn from_hex(value: &[u8]) -> BigUint {
    BigUint::parse_bytes(value, 16).unwrap()
}

fn scalar_modulus() -> BigUint {
    from_hex(b"ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551")
}

fn inverse(value: &BigUint, modulus: &BigUint) -> BigUint {
    let mut t = BigInt::zero();
    let mut new_t = BigInt::one();
    let mut r = BigInt::from(modulus.clone());
    let mut new_r = BigInt::from(value.clone());
    while !new_r.is_zero() {
        let quotient = &r / &new_r;
        (t, new_t) = (new_t.clone(), t - &quotient * new_t);
        (r, new_r) = (new_r.clone(), r - quotient * new_r);
    }
    assert_eq!(r, BigInt::one());
    let modulus = BigInt::from(modulus.clone());
    let mut t = t % &modulus;
    if t.sign() == Sign::Minus {
        t += modulus;
    }
    t.to_biguint().unwrap()
}

fn signature_aux(digest: &BigUint) -> [BigUint; 6] {
    let modulus = scalar_modulus();
    let gx = from_hex(b"6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296");
    let gy = from_hex(b"4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5");
    let s = (digest + &gx) % &modulus;
    [
        gx.clone(),
        gy,
        gx.clone(),
        s.clone(),
        inverse(&gx, &modulus),
        inverse(&s, &modulus),
    ]
}

fn p256_input() -> Box<[bool; VERIFY_DIGEST_INPUT_BITS]> {
    let digest = BigUint::one();
    let aux = signature_aux(&digest);
    let bits: Box<[bool]> = (0..VERIFY_DIGEST_INPUT_BITS)
        .map(|index| {
            if index < 256 {
                digest.bit(index as u64)
            } else {
                let index = index - 256;
                aux[index / 256].bit((index % 256) as u64)
            }
        })
        .collect();
    bits.try_into().unwrap()
}

fn combined_input() -> Box<[bool; VERIFY_2KB_INPUT_BITS]> {
    let digest = from_hex(b"10fc3c51a152e90e5b90319b601d92ccf37290ef53c35ff92507687d8a911a08");
    let aux = signature_aux(&digest);
    let bits: Box<[bool]> = (0..VERIFY_2KB_INPUT_BITS)
        .map(|index| {
            if index < SHA256_2KB_MESSAGE_BITS {
                let byte = (index / 8) as u8;
                byte & (1 << (7 - index % 8)) != 0
            } else {
                let index = index - SHA256_2KB_MESSAGE_BITS;
                aux[index / 256].bit((index % 256) as u64)
            }
        })
        .collect();
    bits.try_into().unwrap()
}

fn dummy_inputs<const N: usize>() -> Box<[Dummy; N]> {
    vec![Dummy; N]
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| unreachable!("dummy input length is fixed"))
}

fn stored_bigint(value: &[u64]) -> BigInt {
    let bytes = value
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    BigInt::from_signed_bytes_le(&bytes)
}

fn reduced_words(value: &BigInt, modulus: &BigUint) -> [u64; 2] {
    let modulus = BigInt::from(modulus.clone());
    let mut value = value % &modulus;
    if value.is_negative() {
        value += modulus;
    }
    let digits = value.to_biguint().unwrap().to_u64_digits();
    let mut words = [0; 2];
    for (output, input) in words.iter_mut().zip(digits) {
        *output = input;
    }
    words
}

/// Independent arbitrary-precision evaluation of each `A/B/C` row against a
/// concrete `Mw`, equivalent to multiplying sparse rows without storing them.
struct DirectAbcProjector<'a> {
    label: &'a str,
    next_integer_witness: usize,
    row: usize,
    integer_witness: &'a PackedWitness,
    exact: &'a IntegerProducts,
    reduced: MatrixProducts<2>,
    modulus: BigUint,
}

impl<'a> DirectAbcProjector<'a> {
    fn new(label: &'a str, witgen: &'a ProductWitgen) -> Self {
        let modulus = (BigUint::one() << 128_usize) - BigUint::from(159_u64);
        let runtime_modulus = ModRingCtx::<2>::new(field::Uint::from_words(
            crate::matrix_products::biguint_words(&(modulus.clone())),
        ))
        .unwrap();
        Self {
            label,
            next_integer_witness: 0,
            row: 0,
            integer_witness: witgen.integer_witness(),
            exact: witgen.products(),
            reduced: witgen.products().reduce_parallel(&runtime_modulus),
            modulus,
        }
    }

    fn finish(self) {
        assert_eq!(
            self.next_integer_witness + 1,
            self.integer_witness.bit_len()
        );
        assert_eq!(self.row, self.exact.a_mw.len());
        assert_eq!(self.row, self.exact.b_mw.len());
        assert_eq!(self.row, self.exact.c_mw.len());
    }
}

impl Circuit for DirectAbcProjector<'_> {
    type Bool = Dummy;
    type Coefficient<const LIMBS: usize> = BigInt;
    type Z<const LIMBS: usize> = BigInt;

    fn coefficient_from_le_words<const LIMBS: usize>(words: &[u64]) -> BigInt {
        let bytes = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        BigInt::from_bytes_le(Sign::Plus, &bytes)
    }

    fn xor(&mut self, _: Dummy, _: Dummy) -> Dummy {
        Dummy
    }

    fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
        &mut self,
        _: H,
    ) -> ScalarBits<Dummy, N>
    where
        H: Fn(&dyn WitnessContext<BigInt, Dummy, BigInt>) -> HintResult<PackedBits<N, M>>
            + Send
            + Sync
            + 'static,
    {
        assert_eq!(M, N.div_ceil(64), "incorrect packed limb count");
        ScalarBits([Dummy; N])
    }

    fn bitz<const LIMBS: usize>(&mut self, _: Dummy) -> BigInt {
        let witness = self.next_integer_witness;
        self.next_integer_witness += 1;
        BigInt::from(self.integer_witness.bit(witness + 1))
    }

    fn assert_r1c<const LIMBS: usize>(
        &mut self,
        expected_a: BigInt,
        expected_b: BigInt,
        expected_c: BigInt,
    ) {
        let row = self.row;
        self.row += 1;
        assert_eq!(
            stored_bigint(&self.exact.a_mw[row]),
            expected_a,
            "{} A row {row}",
            self.label
        );
        assert_eq!(
            stored_bigint(&self.exact.b_mw[row]),
            expected_b,
            "{} B row {row}",
            self.label
        );
        assert_eq!(
            stored_bigint(&self.exact.c_mw[row]),
            expected_c,
            "{} C row {row}",
            self.label
        );
        assert_eq!(
            self.reduced.a_mw.get(row),
            reduced_words(&expected_a, &self.modulus),
            "{} reduced A row {row}",
            self.label
        );
        assert_eq!(
            self.reduced.b_mw.get(row),
            reduced_words(&expected_b, &self.modulus),
            "{} reduced B row {row}",
            self.label
        );
        assert_eq!(
            self.reduced.c_mw.get(row),
            reduced_words(&expected_c, &self.modulus),
            "{} reduced C row {row}",
            self.label
        );
    }

    fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(
        &mut self,
        value: BigInt,
    ) -> BigInt {
        assert!(TO_LIMBS >= FROM_LIMBS);
        value
    }
}

fn projection_challenges(rows: usize) -> Vec<Gf128> {
    (0..rows)
        .map(|index| {
            Gf128::new(
                (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                (index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
            )
        })
        .collect()
}

/// Independent row-oriented multiplication by `M`, using the reference
/// `BTreeSet` Boolean expressions rather than the fast transposed arena.
struct DirectMProjector<'a> {
    challenges: &'a [Gf128],
    next_boolean_witness: usize,
    next_row: usize,
    output: Vec<Gf128>,
    inputs: Box<[BoolLinearCombination]>,
}

impl<'a> DirectMProjector<'a> {
    fn new(input_count: usize, challenges: &'a [Gf128]) -> Self {
        assert!(!challenges.is_empty());
        let inputs = (0..input_count)
            .map(BoolLinearCombination::witness)
            .collect();
        let mut output = vec![Gf128::new(0, 0); input_count + 1];
        output[0] = challenges[0];
        Self {
            challenges,
            next_boolean_witness: input_count,
            next_row: 1,
            output,
            inputs,
        }
    }

    fn take_boxed_inputs<const N: usize>(&mut self) -> Box<[BoolLinearCombination; N]> {
        assert_eq!(N, self.inputs.len());
        std::mem::take(&mut self.inputs)
            .try_into()
            .unwrap_or_else(|_| unreachable!("input length was checked"))
    }

    fn project_row(&mut self, value: &BoolLinearCombination) {
        let challenge = self.challenges[self.next_row];
        self.next_row += 1;
        if value.constant() {
            self.output[0] += challenge;
        }
        for &witness in value.witnesses() {
            self.output[witness + 1] += challenge;
        }
    }

    fn finish(self) -> Vec<Gf128> {
        assert_eq!(self.next_row, self.challenges.len());
        assert_eq!(self.output.len(), self.next_boolean_witness + 1);
        self.output
    }
}

impl Circuit for DirectMProjector<'_> {
    type Bool = BoolLinearCombination;
    type Coefficient<const LIMBS: usize> = Dummy;
    type Z<const LIMBS: usize> = Dummy;

    fn coefficient_from_le_words<const LIMBS: usize>(_: &[u64]) -> Dummy {
        Dummy
    }

    fn xor(
        &mut self,
        lhs: BoolLinearCombination,
        rhs: BoolLinearCombination,
    ) -> BoolLinearCombination {
        lhs.xor(rhs)
    }

    fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
        &mut self,
        _: H,
    ) -> ScalarBits<BoolLinearCombination, N>
    where
        H: Fn(
                &dyn WitnessContext<Dummy, BoolLinearCombination, Dummy>,
            ) -> HintResult<PackedBits<N, M>>
            + Send
            + Sync
            + 'static,
    {
        assert_eq!(M, N.div_ceil(64), "incorrect packed limb count");
        let first = self.next_boolean_witness;
        self.next_boolean_witness += N;
        self.output
            .resize(self.next_boolean_witness + 1, Gf128::new(0, 0));
        ScalarBits(array::from_fn(|index| {
            BoolLinearCombination::witness(first + index)
        }))
    }

    fn bitz<const LIMBS: usize>(&mut self, value: BoolLinearCombination) -> Dummy {
        self.project_row(&value);
        Dummy
    }

    fn assert_r1c<const LIMBS: usize>(&mut self, _: Dummy, _: Dummy, _: Dummy) {}

    fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(&mut self, _: Dummy) -> Dummy {
        assert!(TO_LIMBS >= FROM_LIMBS);
        Dummy
    }
}

fn standalone_p256_projections() {
    let inputs = p256_input();
    let mut witgen =
        ProductWitgen::with_inputs_and_capacity(inputs.as_ref(), VERIFY_DIGEST_WITNESS_BITS);
    verify_digest_circuit(&mut witgen, &inputs);

    let mut abc = DirectAbcProjector::new("P256", &witgen);
    verify_digest_circuit(&mut abc, &dummy_inputs());
    abc.finish();

    let challenges = projection_challenges(VERIFY_DIGEST_INTEGER_WITNESS_BITS);
    let mut direct_m = DirectMProjector::new(VERIFY_DIGEST_INPUT_BITS, &challenges);
    let direct_inputs = direct_m.take_boxed_inputs();
    verify_digest_circuit(&mut direct_m, &direct_inputs);
    let expected_rm = direct_m.finish();

    let mut fast_m = MTransposeGenerator::new(VERIFY_DIGEST_INPUT_BITS);
    let fast_inputs = fast_m.take_boxed_inputs();
    verify_digest_circuit(&mut fast_m, &fast_inputs);
    assert_eq!(fast_m.finish().mul_left(&challenges).unwrap(), expected_rm);
}

fn combined_projections() {
    let inputs = combined_input();
    let mut witgen =
        ProductWitgen::with_inputs_and_capacity(inputs.as_ref(), VERIFY_2KB_WITNESS_BITS);
    verify_2kb_message_circuit(&mut witgen, &inputs);

    let mut abc = DirectAbcProjector::new("SHA256-P256", &witgen);
    verify_2kb_message_circuit(&mut abc, &dummy_inputs());
    abc.finish();

    let challenges = projection_challenges(VERIFY_2KB_INTEGER_WITNESS_BITS);
    let mut direct_m = DirectMProjector::new(VERIFY_2KB_INPUT_BITS, &challenges);
    let direct_inputs = direct_m.take_boxed_inputs();
    verify_2kb_message_circuit(&mut direct_m, &direct_inputs);
    let expected_rm = direct_m.finish();

    let mut fast_m = MTransposeGenerator::new(VERIFY_2KB_INPUT_BITS);
    let fast_inputs = fast_m.take_boxed_inputs();
    verify_2kb_message_circuit(&mut fast_m, &fast_inputs);
    assert_eq!(fast_m.finish().mul_left(&challenges).unwrap(), expected_rm);
}

#[test]
fn ecdsa_fast_projections_match_direct_sparse_matmuls() {
    prepare();
    standalone_p256_projections();
    combined_projections();
}
