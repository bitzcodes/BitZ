//! SHA-256 circuits built from the backend-independent BitZ operations.
//!
//! Words are represented twice: as little-endian F2 bits for Boolean logic and
//! as lifted Z bits for integer linear combinations. This follows Freigen's
//! SHA-256 example, including its doubled `Ch` and `Maj` optimization.

use crate::{BoolRepresentation, BoolWitness, Circuit, HintResult, PackedBits, WitnessContext};
use num_traits::{One, Zero};
use std::array;

/// SHA-256 round constants from FIPS 180-4 section 4.2.2.
pub const ROUND_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 initial chaining value from FIPS 180-4 section 5.3.3.
pub const INITIAL_STATE: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Number of bytes accepted by [`sha256_2kb_circuit`].
pub const SHA256_2KB_MESSAGE_BYTES: usize = 2048;

/// Number of input bits accepted by [`sha256_2kb_circuit`].
pub const SHA256_2KB_MESSAGE_BITS: usize = SHA256_2KB_MESSAGE_BYTES * 8;

/// Boolean witnesses allocated by hints in one compression.
pub const COMPRESSION_HINT_BITS: usize = 48 * 34 + 64 * 70 + 8 * 33;

/// Signed 64-bit Z intermediates are sufficient for every SHA-256 gadget.
pub const SHA256_Z_LIMBS: usize = 1;

/// Total Boolean witness size, including the 2 KiB message inputs.
pub const SHA256_2KB_WITNESS_BITS: usize = block_aligned_witness_bits(SHA256_2KB_MESSAGE_BITS);

/// Total witness bits for a block-aligned SHA-256 message.
pub const fn block_aligned_witness_bits(message_bits: usize) -> usize {
    assert!(
        message_bits.is_multiple_of(512),
        "message must be block-aligned"
    );
    message_bits + (message_bits / 512 + 1) * COMPRESSION_HINT_BITS
}

/// Number of bits in a flattened compression input: one block and one state.
pub const COMPRESSION_INPUT_BITS: usize = 512 + 256;

/// An F2 word whose bits are stored least-significant first.
#[derive(Clone)]
pub struct Word<BW: BoolWitness, const N: usize, const M: usize = 1> {
    bits_le: BW::Repr<N, M>,
}

impl<BW, const N: usize, const M: usize> Word<BW, N, M>
where
    BW: BoolWitness,
{
    /// Constructs a word from little-endian bits.
    pub fn new(bits_le: [BW; N]) -> Self {
        Self {
            bits_le: BW::Repr::from_array(bits_le),
        }
    }

    fn from_repr(bits_le: BW::Repr<N, M>) -> Self {
        Self { bits_le }
    }

    /// Constructs a constant word from the low `N` bits of `value`.
    pub fn constant(value: u64) -> Self {
        Self {
            bits_le: BW::Repr::from_u64(value),
        }
    }

    /// Returns one bit of the word.
    pub fn bit(&self, index: usize) -> BW {
        self.bits_le.bit(index)
    }

    /// Bitwise XOR, represented by addition over F2.
    pub fn xor<CS>(&self, circuit: &mut CS, rhs: &Self) -> Self
    where
        CS: Circuit<Bool = BW>,
    {
        Self {
            bits_le: self.bits_le.xor(circuit, &rhs.bits_le),
        }
    }

    /// Bitwise XOR of three words.
    pub fn xor3<CS>(&self, circuit: &mut CS, second: &Self, third: &Self) -> Self
    where
        CS: Circuit<Bool = BW>,
    {
        self.xor(circuit, second).xor(circuit, third)
    }

    /// Rotates the word right by `amount` bits.
    pub fn rotate_right(&self, amount: usize) -> Self {
        Self {
            bits_le: self.bits_le.rotate_right(amount),
        }
    }

    /// Shifts the word right, filling high bits with zero.
    pub fn shift_right(&self, amount: usize) -> Self {
        Self {
            bits_le: self.bits_le.shift_right(amount),
        }
    }
}

/// A word paired with the Z-side lift of each of its bits.
#[derive(Clone)]
pub struct UInt<ZW, BW: BoolWitness, const N: usize, const M: usize = 1> {
    pub word: Word<BW, N, M>,
    z_values: ZValues<ZW>,
}

#[derive(Clone)]
struct ZValues<ZW> {
    full: ZW,
    low_32: ZW,
}

impl<ZW, BW: BoolWitness, const N: usize, const M: usize> UInt<ZW, BW, N, M> {
    fn int_value(&self) -> ZW
    where
        ZW: Clone,
    {
        self.z_values.full.clone()
    }
}

type ShaZ<CS> = <CS as Circuit>::Z<SHA256_Z_LIMBS>;

type ShaCoefficient<CS> = <CS as Circuit>::Coefficient<SHA256_Z_LIMBS>;

type UInt32<CS> = UInt<ShaZ<CS>, <CS as Circuit>::Bool, 32>;

fn uint_from_word<CS, const N: usize, const M: usize>(
    circuit: &mut CS,
    word: Word<CS::Bool, N, M>,
) -> UInt<ShaZ<CS>, CS::Bool, N, M>
where
    CS: Circuit,
{
    let (full, low_32) = circuit.bitz_unsigned::<SHA256_Z_LIMBS, N, M, 32>(&word.bits_le);
    let z_values = ZValues { full, low_32 };
    UInt { word, z_values }
}

fn uint_constant<CS>(value: u64) -> UInt32<CS>
where
    CS: Circuit,
{
    let word = Word::constant(value);
    let full = ShaZ::<CS>::from(ShaCoefficient::<CS>::from(value));
    let z_values = ZValues {
        low_32: full.clone(),
        full,
    };
    UInt { word, z_values }
}

#[derive(Clone)]
enum DoubledValue<BW: BoolWitness> {
    Choice(Word<BW, 32>, Word<BW, 32>, Word<BW, 32>),
    Majority(Word<BW, 32>, Word<BW, 32>, Word<BW, 32>),
}

impl<BW> DoubledValue<BW>
where
    BW: BoolWitness,
{
    fn evaluate<ZW, C>(&self, context: &dyn WitnessContext<ZW, BW, C>) -> u64 {
        match self {
            Self::Choice(x, y, z) => {
                let x = evaluate_word(context, x) as u32;
                let y = evaluate_word(context, y) as u32;
                let z = evaluate_word(context, z) as u32;
                ((x & y) ^ (!x & z)).into()
            }
            Self::Majority(x, y, z) => {
                let x = evaluate_word(context, x) as u32;
                let y = evaluate_word(context, y) as u32;
                let z = evaluate_word(context, z) as u32;
                ((x & y) ^ (x & z) ^ (y & z)).into()
            }
        }
    }
}

fn evaluate_word<ZW, BW: BoolWitness, C>(
    context: &dyn WitnessContext<ZW, BW, C>,
    word: &Word<BW, 32>,
) -> u64 {
    word.bits_le.evaluate(context).low_u64()
}

fn assert_zero<CS>(circuit: &mut CS, value: ShaZ<CS>)
where
    CS: Circuit,
{
    circuit.assert_r1c::<SHA256_Z_LIMBS>(ShaZ::<CS>::zero(), ShaZ::<CS>::zero(), value);
}

/// Decomposes a sum into `WIDTH` bits, constrains the decomposition, and
/// returns its low 32 bits.
fn sum_32<const WIDTH: usize, const TERMS: usize, CS>(
    circuit: &mut CS,
    terms: [UInt32<CS>; TERMS],
) -> UInt32<CS>
where
    CS: Circuit,
{
    assert!((32..=u64::BITS as usize).contains(&WIDTH));
    assert!(
        TERMS <= 1usize << (WIDTH - 32),
        "sum does not fit its witnessed width"
    );
    let words: [Word<CS::Bool, 32>; TERMS] = array::from_fn(|i| terms[i].word.clone());
    let bits = circuit.hint::<SHA256_Z_LIMBS, WIDTH, 1, _>(move |context| {
        let sum = words
            .iter()
            .map(|word| evaluate_word(context, word))
            .sum::<u64>();
        HintResult::Ok(PackedBits::from_u64(sum))
    });
    let wide = uint_from_word(circuit, Word::from_repr(bits));

    let input_sum = terms
        .into_iter()
        .fold(ShaZ::<CS>::zero(), |sum, term| sum + term.int_value());
    assert_zero(circuit, input_sum - wide.int_value());

    let low_32 = wide.z_values.low_32;
    let z_values = ZValues {
        full: low_32.clone(),
        low_32,
    };
    UInt {
        word: Word::new(array::from_fn(|i| wide.word.bit(i))),
        z_values,
    }
}

/// Freigen's optimized sum: Z expressions for `2*Ch` and `2*Maj` are added to
/// twice the ordinary inputs, then the witnessed quotient by two is constrained.
fn sum_doubled_32<const ORDINARY: usize, const DOUBLED: usize, CS>(
    circuit: &mut CS,
    ordinary: [UInt32<CS>; ORDINARY],
    doubled: [ShaZ<CS>; DOUBLED],
    doubled_values: [DoubledValue<CS::Bool>; DOUBLED],
) -> UInt32<CS>
where
    CS: Circuit,
{
    assert!(ORDINARY + DOUBLED <= 7, "35 bits cannot hold this sum");

    let words: [Word<CS::Bool, 32>; ORDINARY] = array::from_fn(|i| ordinary[i].word.clone());
    let bits = circuit.hint::<SHA256_Z_LIMBS, 35, 1, _>(move |context| {
        let ordinary = words
            .iter()
            .map(|word| evaluate_word(context, word))
            .sum::<u64>();
        let extras = doubled_values
            .iter()
            .map(|value| value.evaluate(context))
            .sum::<u64>();
        let half = ordinary + extras;
        HintResult::Ok(PackedBits::from_u64(half))
    });
    let half = uint_from_word(circuit, Word::from_repr(bits));

    let ordinary = ordinary
        .into_iter()
        .fold(ShaZ::<CS>::zero(), |sum, term| sum + term.int_value());
    let doubled = doubled
        .into_iter()
        .fold(ShaZ::<CS>::zero(), |sum, term| sum + term);
    let mut two = ShaCoefficient::<CS>::one();
    two += ShaCoefficient::<CS>::one();
    let total = ordinary * two.clone() + doubled;
    assert_zero(circuit, total - half.int_value() * two);

    let low_32 = half.z_values.low_32;
    let z_values = ZValues {
        full: low_32.clone(),
        low_32,
    };
    UInt {
        word: Word::new(array::from_fn(|i| half.word.bit(i))),
        z_values,
    }
}

/// Returns a Z expression equal to twice SHA-256's `Ch(x, y, z)`.
fn choice_twice<CS>(circuit: &mut CS, x: &UInt32<CS>, y: &UInt32<CS>, z: &UInt32<CS>) -> ShaZ<CS>
where
    CS: Circuit,
{
    let xy_word = x.word.xor(circuit, &y.word);
    let xy = uint_from_word(circuit, xy_word);
    let xz_word = x.word.xor(circuit, &z.word);
    let xz = uint_from_word(circuit, xz_word);
    y.int_value() + z.int_value() - xy.int_value() + xz.int_value()
}

/// Returns a Z expression equal to twice SHA-256's `Maj(x, y, z)`.
fn majority_twice<CS>(circuit: &mut CS, x: &UInt32<CS>, y: &UInt32<CS>, z: &UInt32<CS>) -> ShaZ<CS>
where
    CS: Circuit,
{
    let xyz_word = x.word.xor3(circuit, &y.word, &z.word);
    let xyz = uint_from_word(circuit, xyz_word);
    x.int_value() + y.int_value() + z.int_value() - xyz.int_value()
}

/// Applies one SHA-256 compression to a 512-bit block and chaining value.
///
/// Both block and state words store their bits least-significant first. The
/// returned words retain both the Boolean and Z representations so callers may
/// inspect the generated relation or feed their Boolean representation into the
/// next compression.
pub fn compress<CS>(
    circuit: &mut CS,
    block: [Word<CS::Bool, 32>; 16],
    state: [Word<CS::Bool, 32>; 8],
) -> [UInt32<CS>; 8]
where
    CS: Circuit,
{
    let state = state.map(|word| uint_from_word(circuit, word));
    let mut schedule: Vec<UInt32<CS>> = Vec::with_capacity(64);
    for word in block {
        schedule.push(uint_from_word(circuit, word));
    }

    for i in 16..64 {
        let word_15 = &schedule[i - 15].word;
        let sigma_0 = word_15.rotate_right(7).xor3(
            circuit,
            &word_15.rotate_right(18),
            &word_15.shift_right(3),
        );
        let sigma_0 = uint_from_word(circuit, sigma_0);

        let word_2 = &schedule[i - 2].word;
        let sigma_1 = word_2.rotate_right(17).xor3(
            circuit,
            &word_2.rotate_right(19),
            &word_2.shift_right(10),
        );
        let sigma_1 = uint_from_word(circuit, sigma_1);

        schedule.push(sum_32::<34, 4, _>(
            circuit,
            [
                schedule[i - 16].clone(),
                sigma_0,
                schedule[i - 7].clone(),
                sigma_1,
            ],
        ));
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state.clone();

    for i in 0..64 {
        let big_sigma_1_word = e.word.rotate_right(6).xor3(
            circuit,
            &e.word.rotate_right(11),
            &e.word.rotate_right(25),
        );
        let big_sigma_1 = uint_from_word(circuit, big_sigma_1_word);
        let choice = choice_twice(circuit, &e, &f, &g);

        let big_sigma_0_word = a.word.rotate_right(2).xor3(
            circuit,
            &a.word.rotate_right(13),
            &a.word.rotate_right(22),
        );
        let big_sigma_0 = uint_from_word(circuit, big_sigma_0_word);
        let majority = majority_twice(circuit, &a, &b, &c);

        let old_a = a;
        let old_b = b;
        let old_c = c;
        let old_d = d;
        let old_e = e;
        let old_f = f;
        let old_g = g;
        let old_h = h;
        let round_constant = uint_constant::<CS>(u64::from(ROUND_CONSTANTS[i]));

        h = old_g.clone();
        g = old_f.clone();
        f = old_e.clone();
        e = sum_doubled_32(
            circuit,
            [
                old_d,
                old_h.clone(),
                big_sigma_1.clone(),
                round_constant.clone(),
                schedule[i].clone(),
            ],
            [choice.clone()],
            [DoubledValue::Choice(
                old_e.word.clone(),
                old_f.word.clone(),
                old_g.word.clone(),
            )],
        );
        d = old_c.clone();
        c = old_b.clone();
        b = old_a.clone();
        a = sum_doubled_32(
            circuit,
            [
                old_h,
                big_sigma_1,
                round_constant,
                schedule[i].clone(),
                big_sigma_0,
            ],
            [choice, majority],
            [
                DoubledValue::Choice(old_e.word.clone(), old_f.word.clone(), old_g.word.clone()),
                DoubledValue::Majority(old_a.word.clone(), old_b.word.clone(), old_c.word.clone()),
            ],
        );
    }

    let working = [a, b, c, d, e, f, g, h];
    array::from_fn(|i| sum_32::<33, 2, _>(circuit, [state[i].clone(), working[i].clone()]))
}

/// Flattened counterpart of [`compress`], matching Freigen's `permCirc'`.
///
/// The first 512 input bits are 16 block words and the final 256 bits are eight
/// state words. Bits within every word, including the output words, are ordered
/// least-significant first.
pub fn compression_circuit<CS>(
    circuit: &mut CS,
    input: &[CS::Bool; COMPRESSION_INPUT_BITS],
) -> [CS::Bool; 256]
where
    CS: Circuit,
{
    let block =
        array::from_fn(|word| Word::new(array::from_fn(|bit| input[word * 32 + bit].clone())));
    let state = array::from_fn(|word| {
        Word::new(array::from_fn(|bit| input[512 + word * 32 + bit].clone()))
    });
    let output = compress(circuit, block, state);
    array::from_fn(|i| output[i / 32].word.bit(i % 32))
}

/// Returns the standard SHA-256 initial state as constant Boolean words.
pub fn initial_state<BW>() -> [Word<BW, 32>; 8]
where
    BW: BoolWitness,
{
    INITIAL_STATE.map(|word| Word::constant(u64::from(word)))
}

/// Computes SHA-256 for an exactly 2 KiB message.
///
/// Input and output bits use conventional stream order: bytes are ordered from
/// first to last and bits within each byte are most-significant first. Because
/// the input is large, callers should normally own it as
/// `Box<[CS::Bool; SHA256_2KB_MESSAGE_BITS]>`; borrowing it here does not copy
/// the array or move it onto the stack.
pub fn sha256_2kb_circuit<CS>(
    circuit: &mut CS,
    message: &[CS::Bool; SHA256_2KB_MESSAGE_BITS],
) -> [CS::Bool; 256]
where
    CS: Circuit,
{
    sha256_block_aligned_circuit(circuit, SHA256_2KB_MESSAGE_BITS, |bit| message[bit].clone())
}

/// Computes SHA-256 for a block-aligned message supplied by a bit getter.
///
/// Input and output use conventional stream order. `message_bits` must be a
/// multiple of 512; the function adds the final SHA-256 padding block.
pub fn sha256_block_aligned_circuit<CS, F>(
    circuit: &mut CS,
    message_bits: usize,
    message_bit: F,
) -> [CS::Bool; 256]
where
    CS: Circuit,
    F: Fn(usize) -> CS::Bool,
{
    assert!(message_bits.is_multiple_of(512));
    let encoded_length = u64::try_from(message_bits).expect("SHA-256 message length exceeds u64");
    let mut state = initial_state();

    for block_index in 0..message_bits / 512 {
        let block = array::from_fn(|word| {
            Word::new(array::from_fn(|word_bit| {
                let stream_bit = block_index * 512 + word * 32 + (31 - word_bit);
                message_bit(stream_bit)
            }))
        });
        state = compress(circuit, block, state).map(|word| word.word);
    }

    let padding_values: [u32; 16] = [
        0x80000000,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        (encoded_length >> 32) as u32,
        encoded_length as u32,
    ];
    let padding = padding_values.map(|word| Word::constant(u64::from(word)));
    let digest = compress(circuit, padding, state);

    array::from_fn(|i| digest[i / 32].word.bit(31 - (i % 32)))
}

#[cfg(test)]
mod tests {
    use std::iter::Sum;
    use std::ops::{Add, AddAssign};

    use num_traits::Zero;

    use super::*;
    use crate::HintError;
    use crate::constraints::{ConstraintGenerator, ConstraintMatrices};
    use crate::stats::{Dummy, LeanStats, Stats};
    use crate::witgen::{Witgen, Z};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Bit(bool);

    impl From<bool> for Bit {
        fn from(value: bool) -> Self {
            Self(value)
        }
    }

    impl Zero for Bit {
        fn zero() -> Self {
            Self(false)
        }

        fn is_zero(&self) -> bool {
            !self.0
        }
    }

    impl Add for Bit {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 != rhs.0)
        }
    }

    impl AddAssign for Bit {
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl Sum for Bit {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self(false), Add::add)
        }
    }

    impl BoolWitness for Bit {
        type Repr<const N: usize, const M: usize> = crate::ScalarBits<Self, N>;
    }

    struct Values;

    impl WitnessContext<i128, Bit, i128> for Values {
        fn eval_z(&self, witness: &i128) -> i128 {
            *witness
        }

        fn eval_bool(&self, witness: &Bit) -> bool {
            witness.0
        }
    }

    #[derive(Default)]
    struct EvaluatingCircuit {
        assertions: usize,
    }

    impl Circuit for EvaluatingCircuit {
        type Bool = Bit;
        type Coefficient<const LIMBS: usize> = i128;
        type Z<const LIMBS: usize> = i128;

        fn xor(&mut self, lhs: Bit, rhs: Bit) -> Bit {
            lhs + rhs
        }

        fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
            &mut self,
            hint: H,
        ) -> crate::ScalarBits<Bit, N>
        where
            H: Fn(&dyn WitnessContext<i128, Bit, i128>) -> Result<PackedBits<N, M>, HintError>
                + Send
                + Sync
                + 'static,
        {
            crate::ScalarBits::from_packed(hint(&Values).expect("SHA-256 hint should be defined"))
        }

        fn bitz<const LIMBS: usize>(&mut self, value: Bit) -> i128 {
            i128::from(value.0)
        }

        fn assert_r1c<const LIMBS: usize>(&mut self, a: i128, b: i128, c: i128) {
            self.assertions += 1;
            assert_eq!(a * b, c, "unsatisfied SHA-256 constraint");
        }

        fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(
            &mut self,
            value: i128,
        ) -> i128 {
            value
        }
    }

    fn value(word: &Word<Bit, 32>) -> u32 {
        word.bits_le
            .0
            .iter()
            .enumerate()
            .fold(0, |value, (bit, witness)| {
                value | (u32::from(witness.0) << bit)
            })
    }

    fn hash_then_widen<CS>(
        circuit: &mut CS,
        message: &[CS::Bool; SHA256_2KB_MESSAGE_BITS],
    ) -> CS::Z<128>
    where
        CS: Circuit,
    {
        let digest = sha256_2kb_circuit(circuit, message);
        let small = circuit.bitz::<SHA256_Z_LIMBS>(digest[0].clone());
        circuit.sign_extend_z::<SHA256_Z_LIMBS, 128>(small)
    }

    fn assert_m_w_matches_witgen(matrices: &ConstraintMatrices, witgen: &Witgen) {
        let from_m = matrices
            .integer_witness(witgen.witness())
            .expect("Boolean witness should have the matrix width");
        let recorded = witgen.integer_witness();
        assert_eq!(from_m.len(), recorded.bit_len());
        for (row, value) in from_m.iter().enumerate() {
            assert_eq!(
                value,
                &recorded.bit(row),
                "integer witness differs at M row {row}"
            );
        }
    }

    #[test]
    fn compression_matches_the_fips_abc_vector() {
        let block_values: [u32; 16] = [
            0x61626380, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00000018,
        ];
        let block = block_values.map(|word| Word::constant(u64::from(word)));
        let mut circuit = EvaluatingCircuit::default();

        let output = compress(&mut circuit, block, initial_state());
        let output = output.map(|word| value(&word.word));

        assert_eq!(
            output,
            [
                0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
                0xf20015ad,
            ]
        );
        assert_eq!(circuit.assertions, 184);
    }

    #[test]
    fn fixed_2kb_circuit_matches_sha256_stream_order() {
        let message: Box<[Bit]> = (0..SHA256_2KB_MESSAGE_BITS)
            .map(|bit| {
                let byte = (bit / 8) as u8;
                Bit(byte & (1 << (7 - bit % 8)) != 0)
            })
            .collect();
        let message: Box<[Bit; SHA256_2KB_MESSAGE_BITS]> = message
            .try_into()
            .unwrap_or_else(|_| unreachable!("message length is fixed"));
        let mut circuit = EvaluatingCircuit::default();

        let digest_bits = sha256_2kb_circuit(&mut circuit, &message);
        let digest: [u8; 32] = array::from_fn(|byte| {
            (0..8).fold(0, |value, bit| {
                value | (u8::from(digest_bits[byte * 8 + bit].0) << (7 - bit))
            })
        });

        assert_eq!(
            digest,
            [
                0x10, 0xfc, 0x3c, 0x51, 0xa1, 0x52, 0xe9, 0x0e, 0x5b, 0x90, 0x31, 0x9b, 0x60, 0x1d,
                0x92, 0xcc, 0xf3, 0x72, 0x90, 0xef, 0x53, 0xc3, 0x5f, 0xf9, 0x25, 0x07, 0x68, 0x7d,
                0x8a, 0x91, 0x1a, 0x08,
            ]
        );
        assert_eq!(circuit.assertions, 33 * 184);

        let message: Box<[bool]> = message.iter().map(|bit| bit.0).collect();
        let message: Box<[bool; SHA256_2KB_MESSAGE_BITS]> = message
            .try_into()
            .unwrap_or_else(|_| unreachable!("message length is fixed"));
        let mut witgen =
            Witgen::with_inputs_and_capacity(message.as_ref(), SHA256_2KB_WITNESS_BITS);
        let witgen_bits = sha256_2kb_circuit(&mut witgen, &message);
        let witgen_digest: [u8; 32] = array::from_fn(|byte| {
            (0..8).fold(0, |value, bit| {
                value | (u8::from(witgen_bits[byte * 8 + bit]) << (7 - bit))
            })
        });
        assert_eq!(witgen_digest, digest);
        assert_eq!(witgen.witness().bit_len(), SHA256_2KB_WITNESS_BITS);
        assert_eq!(
            witgen.witness().words().len(),
            SHA256_2KB_WITNESS_BITS.div_ceil(64)
        );
        assert!(
            message
                .iter()
                .enumerate()
                .all(|(index, expected)| witgen.witness().bit(index) == *expected)
        );

        let mut generator = ConstraintGenerator::new(SHA256_2KB_MESSAGE_BITS);
        let symbolic_message = generator.boxed_inputs();
        let _ = sha256_2kb_circuit(&mut generator, &symbolic_message);
        let matrices = generator.into_matrices();
        let compressions = SHA256_2KB_MESSAGE_BITS / 512 + 1;
        assert_eq!(matrices.m.row_count(), compressions * 20_456 + 1);
        assert_eq!(matrices.m.column_count(), SHA256_2KB_WITNESS_BITS + 1);
        assert_eq!(matrices.a.row_count(), compressions * 184);
        assert_eq!(matrices.a.column_count(), matrices.m.row_count());
        assert_m_w_matches_witgen(&matrices, &witgen);
        matrices
            .check_witness(witgen.witness())
            .expect("SHA-256 witness should satisfy M/A/B/C");

        let mut composed =
            Witgen::with_inputs_and_capacity(message.as_ref(), SHA256_2KB_WITNESS_BITS);
        let wide: Z<128> = hash_then_widen(&mut composed, &message);
        assert_eq!(wide, Z::<128>::zero());
    }

    #[test]
    fn compression_layout_exactly_matches_freigen() {
        let mut stats = Stats::new(COMPRESSION_INPUT_BITS);

        let _ = compression_circuit(&mut stats, &[Dummy; COMPRESSION_INPUT_BITS]);

        assert_eq!(
            stats,
            Stats {
                witnesses: 7_144,
                bitz_calls: 20_456,
                constraints: 184,
            }
        );
        assert_eq!(
            stats.lean_stats(),
            LeanStats {
                m_rows: 20_457,
                m_cols: 7_145,
                r1cs_rows: 184,
            }
        );
    }

    #[test]
    fn compression_witness_satisfies_materialized_matrices() {
        let inputs: [bool; COMPRESSION_INPUT_BITS] =
            array::from_fn(|bit| bit % 7 == 1 || bit % 13 == 4);
        let mut witgen = Witgen::with_inputs_and_capacity(
            &inputs,
            COMPRESSION_INPUT_BITS + COMPRESSION_HINT_BITS,
        );
        let _ = compression_circuit(&mut witgen, &inputs);

        let mut generator = ConstraintGenerator::new(COMPRESSION_INPUT_BITS);
        let symbolic_inputs = generator.inputs();
        let _ = compression_circuit(&mut generator, &symbolic_inputs);
        let matrices = generator.into_matrices();

        assert_eq!(matrices.m.row_count(), 20_457);
        assert_eq!(matrices.m.column_count(), 7_145);
        assert_eq!(matrices.a.row_count(), 184);
        assert_eq!(matrices.b.row_count(), 184);
        assert_eq!(matrices.c.row_count(), 184);
        assert_eq!(matrices.a.column_count(), 20_457);
        assert_m_w_matches_witgen(&matrices, &witgen);
        matrices
            .check_witness(witgen.witness())
            .expect("SHA-256 compression witness should satisfy M/A/B/C");
    }
}
