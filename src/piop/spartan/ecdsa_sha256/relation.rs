use circuit::linear_map::CscMatrix;
use std::{
    array,
    collections::HashMap,
    sync::{Arc, OnceLock},
};

use circuit::integer_storage::IntegerTable;
use circuit::linear_map::{CsrBuilder, CsrMatrix, IndexedCoefficients};
use circuit::{
    constraints::ConstraintGenerator,
    linear_map::circuit::{WengertGenerator, WengertTape},
    p256, sha256,
};
use field::{CtOrd, IntegerOps, Uint, WideMul, ZRef};

use super::error;
use {
    crate::pcs::IntegerMatrixLayout,
    circuit::linear_map::binary::{
        ChainedPackedSourceParts, ChainedSourceTail, PreparedVirtualMap, VirtualMap,
    },
};

pub(crate) const SHA_H: usize = 20_457;
pub(crate) const SHA_F: usize = 512 + sha256::COMPRESSION_HINT_BITS;
pub(crate) const SHA_ROWS: usize = 184;
pub(crate) const P_INPUT_ALIAS: usize = 257;

/// Which original circuit rows participate in the outer sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuterMode {
    Split,
    AllRows,
}

/// Public values use fixed-width big-endian encodings. The message is a witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sha256EcdsaStatement {
    pub log_compressions: u8,
    pub qx: [u8; 32],
    pub qy: [u8; 32],
    pub r: [u8; 32],
    pub s: [u8; 32],
}

impl Sha256EcdsaStatement {
    pub fn message_bytes(&self) -> Result<usize, super::Sha256EcdsaError> {
        if !(3..=16).contains(&self.log_compressions) {
            return Err(error("compression exponent must be in 3..=16"));
        }
        Ok(64 * ((1usize << self.log_compressions) - 1))
    }
    pub(crate) fn bytes(&self) -> Vec<u8> {
        let mut bytes = vec![self.log_compressions];
        for word in [&self.qx, &self.qy, &self.r, &self.s] {
            bytes.extend_from_slice(word);
        }
        bytes
    }
    pub(crate) fn bit(&self, bit: usize) -> bool {
        let words = [&self.qx, &self.qy, &self.r, &self.s];
        words[bit / 256][31 - (bit % 256) / 8] >> (bit % 8) & 1 != 0
    }
}

/// Interns matrix coefficients into the shared table, first seen first.
#[derive(Default)]
struct Interner {
    table: IntegerTable,
    index: HashMap<Vec<u64>, u32>,
}

impl Interner {
    fn intern(&mut self, matrix: &CsrMatrix<IntegerTable>, coefficient: usize) -> u32 {
        let words = &matrix.coefficients()[coefficient];
        if let Some(&k) = self.index.get(words) {
            return k;
        }
        let k = u32::try_from(self.table.len()).expect("coefficient table fits u32");
        matrix
            .coefficients()
            .copy_row_to(coefficient, &mut self.table);
        self.index.insert(words.to_vec(), k);
        k
    }
    fn rows(&mut self, matrix: &CsrMatrix<IntegerTable>) -> CsrMatrix<Box<[u32]>, u32> {
        let mut builder = CsrBuilder::<Vec<u32>, u32>::new(Vec::new());
        for row in matrix.rows() {
            builder
                .push_row(
                    row.indices()
                        .iter()
                        .zip(row.entry_range())
                        .map(|(&column, entry)| (column, self.intern(matrix, entry))),
                )
                .expect("canonical compact circuit rows");
        }
        let matrix = builder
            .finish(matrix.column_count())
            .expect("allocated columns");
        matrix
            .map_coefficients(Vec::into_boxed_slice)
            .expect("unchanged entry count")
    }
}

/// Interleave A/B/C rows before CSC conversion so encoded slots remain sorted.
fn tail_columns(
    columns: usize,
    matrices: [&CsrMatrix<Box<[u32]>, u32>; 3],
) -> Result<CscMatrix<Box<[u32]>, u32>, super::Sha256EcdsaError> {
    CscMatrix::try_from_row_source(3 * matrices[0].row_count(), columns, |slot| {
        matrices[slot % 3].row(slot / 3).unwrap()
    })
    .map_err(|_| error("invalid compact P-256 tail matrix"))
}

pub(crate) struct LocalRelation {
    pub sha_local: PreparedVirtualMap,
    pub sha_prev: PreparedVirtualMap,
    pub sha_first: PreparedVirtualMap,
    pub sha_c: CsrMatrix<IndexedCoefficients<Arc<IntegerTable>>, u32>,
    pub sha_output: [usize; 256],
    pub p_map: PreparedVirtualMap,
    pub a: CsrMatrix<IndexedCoefficients<Arc<IntegerTable>>, u32>,
    pub b: CsrMatrix<IndexedCoefficients<Arc<IntegerTable>>, u32>,
    pub c: CsrMatrix<IndexedCoefficients<Arc<IntegerTable>>, u32>,
    /// The distinct integer coefficients of `sha_c`, `a`, `b` and `c`.
    pub coefficients: Arc<IntegerTable>,
    /// `a`, `b`, `c` column by column over the P-256 assignment tail.
    pub tail: CscMatrix<IndexedCoefficients<Arc<IntegerTable>>, u32>,
    /// The P-256 circuit's Z-side linear arithmetic as a reverse-mode tape:
    /// `r · (A + xB + x²C)` over every tail column by one pass over the
    /// circuit's DAG (97,986 edges) instead of the 4.2M expanded entries.
    pub tape: WengertTape,
    pub nonlinear: Vec<usize>,
    pub linear: Vec<usize>,
    pub public_h: [usize; 1024],
    pub digest: [u8; 32],
    pub defect_bits: u32,
}

impl LocalRelation {
    /// Number of P-256 rows (shared by `a`, `b` and `c`).
    pub(crate) fn rows(&self) -> usize {
        self.a.row_count()
    }
}

fn bool_map(
    columns: usize,
    rows: Vec<Vec<usize>>,
) -> Result<PreparedVirtualMap, super::Sha256EcdsaError> {
    let rows = rows
        .into_iter()
        .map(|row| row.into_iter().map(|c| (c, true)).collect())
        .collect();
    PreparedVirtualMap::new(CscMatrix::try_from_rows(columns, rows).map_err(error)?).map_err(error)
}

fn build_local() -> Result<LocalRelation, super::Sha256EcdsaError> {
    let mut generator = ConstraintGenerator::new(sha256::COMPRESSION_INPUT_BITS);
    let inputs = generator.inputs();
    let outputs = sha256::compression_circuit(&mut generator, &inputs);
    let sha_output = array::from_fn(|i| {
        assert!(!outputs[i].constant() && outputs[i].witnesses().len() == 1);
        // The chained source omits the independent compression's state inputs.
        outputs[i].witnesses().first().copied().unwrap() + 1 - 256
    });
    let sha = generator.into_matrices();
    if sha.m.row_count() != SHA_H
        || sha.c.row_count() != SHA_ROWS
        || sha.a.rows().chain(sha.b.rows()).any(|r| !r.is_empty())
    {
        return Err(error("unexpected SHA compression shape"));
    }
    let mut local = vec![Vec::new(); SHA_H];
    let mut prev = vec![Vec::new(); SHA_H];
    let mut first = vec![Vec::new(); SHA_H];
    for (row, terms) in sha.m.rows().enumerate() {
        let mut parity = false;
        for &column in terms.indices() {
            match column {
                0..=512 => local[row].push(column),
                513..=768 => {
                    let bit = column - 513;
                    prev[row].push(sha_output[bit]);
                    parity ^= sha256::INITIAL_STATE[bit / 32] >> (bit % 32) & 1 != 0;
                }
                _ => local[row].push(column - 256),
            }
        }
        if parity {
            first[row].push(0);
        }
    }
    let sha_local = bool_map(SHA_F + 1, local)?;
    let sha_prev = bool_map(SHA_F + 1, prev)?;
    let sha_first = bool_map(SHA_F + 1, first)?;
    let mut interner = Interner::default();
    let sha_c = interner.rows(&sha.c);
    drop(sha);

    let mut generator = ConstraintGenerator::new(p256::VERIFY_DIGEST_INPUT_BITS);
    let inputs = generator.inputs();
    p256::verify_digest_circuit(&mut generator, &inputs);
    let p = generator.into_matrices();
    let mut public_h = [usize::MAX; 1024];
    for (h, row) in p.m.rows().enumerate() {
        if let [source] = row.indices() {
            if (257..1281).contains(source) && public_h[*source - 257] == usize::MAX {
                public_h[*source - 257] = h;
            }
        }
    }
    if public_h.contains(&usize::MAX) {
        return Err(error("public P-256 inputs have no direct lifts"));
    }
    let p_map = bool_map(
        p.m.column_count(),
        p.m.rows().map(|r| r.indices().to_vec()).collect(),
    )?;
    let a = interner.rows(&p.a);
    let b = interner.rows(&p.b);
    let c = interner.rows(&p.c);
    let tail = tail_columns(p.m.row_count(), [&a, &b, &c])?;
    drop(p);
    let mut tape_generator = WengertGenerator::new(p256::VERIFY_DIGEST_INPUT_BITS);
    let tape_inputs = tape_generator.take_boxed_inputs::<{ p256::VERIFY_DIGEST_INPUT_BITS }>();
    p256::verify_digest_circuit(&mut tape_generator, &tape_inputs);
    let tape = tape_generator.finish();
    if tape.row_count() != a.row_count() || tape.column_count() != tail.column_count() {
        return Err(error(
            "P-256 tape shape disagrees with the constraint matrices",
        ));
    }
    let coefficients = Arc::new(interner.table);
    let attach = |matrix: CsrMatrix<Box<[u32]>, u32>| {
        matrix
            .map_coefficients(|indices| {
                IndexedCoefficients::new(Arc::clone(&coefficients), indices.into_vec())
                    .expect("interned coefficients")
            })
            .expect("unchanged entry count")
    };
    let sha_c = attach(sha_c);
    let a = attach(a);
    let b = attach(b);
    let c = attach(c);
    let tail = tail
        .map_coefficients(|indices| {
            IndexedCoefficients::new(Arc::clone(&coefficients), indices.into_vec())
                .expect("interned coefficients")
        })
        .expect("unchanged entry count");
    let (linear, nonlinear): (Vec<_>, Vec<_>) = (0..a.row_count())
        .partition(|&i| a.row(i).unwrap().is_empty() || b.row(i).unwrap().is_empty());
    // P-256 declares at most nine signed limbs. One extra limb covers a
    // row with fewer than 2^64 terms; the product of two row norms plus C
    // fits 21 limbs. These public bounds never depend on witness magnitudes.
    if coefficients.max_limbs() > 9 {
        return Err(error("P-256 coefficient declaration exceeds nine limbs"));
    }
    let magnitudes: Vec<Uint<10>> = coefficients
        .iter()
        .map(|words| {
            ZRef::from_twos_complement_words(words)
                .checked_resize_ct::<9>()
                .value()
                .unsigned_abs()
                .zero_extend()
        })
        .collect();
    let norm =
        |rows: &CsrMatrix<IndexedCoefficients<Arc<IntegerTable>>, u32>, r: usize| -> Uint<10> {
            rows.row(r)
                .unwrap()
                .indexed_entries()
                .fold(Uint::ZERO, |sum, (_, k)| sum.wrapping_add(&magnitudes[k]))
        };
    let mut bound = Uint::<21>::ONE;
    for r in 0..sha_c.row_count() {
        let candidate = norm(&sha_c, r).zero_extend::<21>();
        if bound.ct_lt(&candidate).declassify() {
            bound = candidate;
        }
    }
    // The assignment is Boolean, so the public coefficient 1-norm bounds
    // every row operand, including malformed witnesses. Retaining this check
    // lets the borrowed outer rows use five signed limbs for A/B without
    // inspecting their private magnitudes or allocating narrowed tables.
    let outer_ab_limit = Uint::<10>::from_words([0, 0, 0, 0, 1 << 63, 0, 0, 0, 0, 0]);
    for i in 0..a.row_count() {
        let a_norm = norm(&a, i);
        let b_norm = norm(&b, i);
        if !(a_norm.ct_lt(&outer_ab_limit) & b_norm.ct_lt(&outer_ab_limit)).declassify() {
            return Err(error("P-256 A/B row bound exceeds five signed limbs"));
        }
        let product = IntegerOps.mul_wide(&a_norm, &b_norm);
        let candidate = product
            .checked_resize_ct::<21>()
            .value()
            .wrapping_add(&norm(&c, i).zero_extend());
        if bound.ct_lt(&candidate).declassify() {
            bound = candidate;
        }
    }
    let defect_bits = bound
        .as_words()
        .iter()
        .enumerate()
        .rev()
        .find(|(_, word)| **word != 0)
        .map_or(0, |(i, word)| (64 * i) as u32 + 64 - word.leading_zeros());
    // The digest binds each declared signed width and every encoded limb.
    let signed_bytes: Vec<Vec<u8>> = coefficients
        .iter()
        .map(|words| words.iter().flat_map(|word| word.to_le_bytes()).collect())
        .collect();
    let mut hash = blake3::Hasher::new();
    hash.update(b"bitz/sha256-ecdsa/local-relation/v2");
    for map in [&sha_local, &sha_prev, &sha_first, &p_map] {
        hash.update(&map.digest());
    }
    let mut buffer = Vec::new();
    for matrix in [&sha_c, &a, &b, &c] {
        hash.update(&(matrix.row_count() as u64).to_le_bytes());
        for r in 0..matrix.row_count() {
            buffer.clear();
            buffer.extend_from_slice(
                &((matrix.row_offsets()[r + 1] - matrix.row_offsets()[r]) as u64).to_le_bytes(),
            );
            for (column, k) in matrix.row(r).unwrap().indexed_entries() {
                buffer.extend_from_slice(&(column as u64).to_le_bytes());
                buffer.extend_from_slice(&(signed_bytes[k].len() as u64).to_le_bytes());
                buffer.extend_from_slice(&signed_bytes[k]);
            }
            hash.update(&buffer);
        }
    }
    Ok(LocalRelation {
        sha_local,
        sha_prev,
        sha_first,
        sha_c,
        sha_output,
        p_map,
        a,
        b,
        c,
        coefficients,
        tail,
        tape,
        nonlinear,
        linear,
        public_h,
        digest: *hash.finalize().as_bytes(),
        defect_bits,
    })
}

/// One implicit chained SHA map followed by a compact P-256 map.
pub struct Sha256EcdsaMap {
    pub(crate) local: Arc<LocalRelation>,
    pub(crate) last: PreparedVirtualMap,
    pub(crate) n: usize,
    pub(crate) h_offset: usize,
    pub(crate) f_offset: usize,
    aliases: [usize; P_INPUT_ALIAS],
    rows: usize,
    cols: usize,
    nnz: usize,
    digest: [u8; 32],
}

impl Sha256EcdsaMap {
    pub(crate) fn p_source(&self, column: usize) -> usize {
        match column {
            0 => 0,
            1..=256 => {
                let digest_bit = column - 1;
                let bit = (7 - digest_bit / 32) * 32 + digest_bit % 32;
                (self.n - 1) * SHA_F + self.local.sha_output[bit]
            }
            _ => self.f_offset + column - P_INPUT_ALIAS,
        }
    }
}

impl VirtualMap for Sha256EcdsaMap {
    type ColumnRows<'a> = std::vec::IntoIter<usize>;
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn nnz(&self) -> usize {
        self.nnz
    }
    fn digest(&self) -> [u8; 32] {
        self.digest
    }
    fn is_identity(&self) -> bool {
        false
    }
    fn chained_packed_source(&self) -> Option<ChainedPackedSourceParts<'_>> {
        Some(ChainedPackedSourceParts {
            local: &self.local.sha_local,
            prev: &self.local.sha_prev,
            first: &self.local.sha_first,
            last: &self.last,
            instances: self.n,
        })
    }
    fn chained_packed_source_tail(&self) -> Option<ChainedSourceTail<'_>> {
        Some(ChainedSourceTail {
            map: &self.local.p_map,
            row_offset: self.h_offset,
            source_offset: self.f_offset,
            aliases: &self.aliases,
        })
    }
    fn column_rows(&self, column: usize) -> Option<Self::ColumnRows<'_>> {
        if column >= self.cols {
            return None;
        }
        let mut rows = Vec::new();
        let mut add = |map: &PreparedVirtualMap, c: usize, instance: usize| {
            if let Some(col) = map.matrix().column(c) {
                rows.extend(col.indices().iter().map(|r| instance + self.n * r));
            }
        };
        if column == 0 {
            for i in 0..self.n {
                add(&self.local.sha_local, 0, i);
            }
            add(&self.local.sha_first, 0, 0);
            add(&self.last, 0, self.n - 1);
        } else if column < self.f_offset {
            let instance = (column - 1) / SHA_F;
            let c = (column - 1) % SHA_F + 1;
            add(&self.local.sha_local, c, instance);
            if instance + 1 < self.n {
                add(&self.local.sha_prev, c, instance + 1);
            }
            if instance + 1 == self.n {
                add(&self.last, c, instance);
            }
        }
        let mut add_p = |c: usize| {
            if let Some(col) = self.local.p_map.matrix().column(c) {
                rows.extend(col.indices().iter().map(|r| self.h_offset + r));
            }
        };
        if column == 0 {
            add_p(0);
        } else if column >= self.f_offset {
            add_p(P_INPUT_ALIAS + column - self.f_offset);
        } else if column > (self.n - 1) * SHA_F {
            for c in 1..=256 {
                if self.p_source(c) == column {
                    add_p(c);
                }
            }
        }
        rows.sort_unstable();
        let mut parity = Vec::with_capacity(rows.len());
        for row in rows {
            if parity.last() == Some(&row) {
                parity.pop();
            } else {
                parity.push(row);
            }
        }
        Some(parity.into_iter())
    }
}

/// Prepared public relation. No message or signature is retained here.
pub struct PreparedSha256Ecdsa {
    pub(crate) local: Arc<LocalRelation>,
    pub(crate) map: Sha256EcdsaMap,
    pub(crate) log_n: usize,
    pub(crate) mode: OuterMode,
    pub(crate) lambda: u32,
    pub(crate) h_layout: IntegerMatrixLayout,
    pub(crate) f_layout: IntegerMatrixLayout,
    pub(crate) ligerito: crate::ligerito_flock::ResolvedLigerito,
}

impl PreparedSha256Ecdsa {
    pub fn ligerito_configuration(&self) -> &crate::ligerito_flock::ResolvedLigerito {
        &self.ligerito
    }

    pub fn with_ligerito(
        mut self,
        selection: crate::ligerito_flock::LigeritoSelection,
    ) -> Result<Self, super::Sha256EcdsaError> {
        self.ligerito = selection
            .resolve(
                self.f_layout.row_vars + self.f_layout.col_vars - 7,
                self.lambda as usize,
            )
            .map_err(error)?;
        self.security()?;
        Ok(self)
    }

    pub fn compressions(&self) -> usize {
        self.map.n
    }
    pub fn message_bytes(&self) -> usize {
        64 * (self.compressions() - 1)
    }
    pub fn nonlinear_rows(&self) -> usize {
        self.local.nonlinear.len()
    }
    /// Original rows entering the outer sumcheck in the selected mode.
    pub fn outer_rows(&self) -> usize {
        match self.mode {
            OuterMode::Split => self.local.nonlinear.len(),
            OuterMode::AllRows => SHA_ROWS * self.compressions() + self.local.rows(),
        }
    }
    /// Allocated outer table slots, including layout and power-of-two padding.
    pub fn outer_domain_size(&self) -> usize {
        1 << self.outer_sumcheck_num_vars()
    }
    pub(super) fn outer_sumcheck_num_vars(&self) -> usize {
        let rows = match self.mode {
            OuterMode::Split => self.local.nonlinear.len(),
            OuterMode::AllRows => 256 * self.compressions() + self.local.rows(),
        };
        rows.next_power_of_two().ilog2() as usize
    }
    pub fn linear_rows(&self) -> usize {
        self.compressions() * SHA_ROWS + self.local.linear.len() + 1025
    }
    pub fn live_assignment_bits(&self) -> usize {
        self.map.h_offset + self.local.p_map.rows()
    }
    pub fn live_source_bits(&self) -> usize {
        self.map.f_offset + self.local.p_map.cols() - P_INPUT_ALIAS
    }
    pub fn assignment_params(&self) -> &IntegerMatrixLayout {
        &self.h_layout
    }
    pub fn source_params(&self) -> &IntegerMatrixLayout {
        &self.f_layout
    }
    pub fn map(&self) -> &impl VirtualMap {
        &self.map
    }
    pub fn security_target(&self) -> u32 {
        self.lambda
    }
    pub fn outer_mode(&self) -> OuterMode {
        self.mode
    }
    pub(crate) fn padding(&self) -> [u32; 16] {
        padding(self.compressions())
    }
    pub(crate) fn linear_vars(&self) -> usize {
        (256 * self.compressions() + self.local.linear.len() + 1025)
            .next_power_of_two()
            .ilog2() as usize
    }
}

pub(crate) fn padding(n: usize) -> [u32; 16] {
    let mut words = [0u32; 16];
    words[0] = 0x8000_0000;
    let bits = ((n - 1) * 512) as u64;
    words[14] = (bits >> 32) as u32;
    words[15] = bits as u32;
    words
}

pub fn prepare_sha256_ecdsa(
    log_compressions: usize,
    lambda: u32,
    mode: OuterMode,
) -> Result<PreparedSha256Ecdsa, super::Sha256EcdsaError> {
    if !(3..=16).contains(&log_compressions) || ![100, 128].contains(&lambda) {
        return Err(error("expected exponent 3..=16 and security 100 or 128"));
    }
    static LOCAL: OnceLock<std::result::Result<Arc<LocalRelation>, String>> = OnceLock::new();
    let local = LOCAL
        .get_or_init(|| build_local().map(Arc::new).map_err(|e| e.to_string()))
        .as_ref()
        .map_err(error)?
        .clone();
    let n = 1usize << log_compressions;
    let h_offset = SHA_H * n;
    let f_offset = 1 + SHA_F * n;
    let h_bits = (h_offset + local.p_map.rows()).next_power_of_two().ilog2() as usize;
    let f_bits = (f_offset + local.p_map.cols() - P_INPUT_ALIAS)
        .next_power_of_two()
        .ilog2() as usize;
    let params = |bits: usize| {
        let t = bits.div_ceil(2).min(13);
        IntegerMatrixLayout {
            row_vars: t,
            col_vars: bits - t,
            word_bits: 1,
        }
    };
    // At the final block cancel its source block inputs and replace them by
    // the mandated padding bits, using the shared source-one coordinate.
    let mut last = vec![Vec::new(); SHA_H];
    let pad = padding(n);
    let mut canceled = 0;
    for bit in 0..512 {
        for &row in local.sha_local.matrix().column(bit + 1).unwrap().indices() {
            last[row].push(bit + 1);
            canceled += 1;
            if pad[bit / 32] >> (bit % 32) & 1 != 0 {
                last[row].push(0);
            }
        }
    }
    for row in &mut last {
        row.sort_unstable();
        let mut out = Vec::new();
        for &c in row.iter() {
            if out.last() == Some(&c) {
                out.pop();
            } else {
                out.push(c);
            }
        }
        *row = out;
    }
    let last = bool_map(SHA_F + 1, last)?;
    let nnz = n * local.sha_local.nnz()
        + (n - 1) * local.sha_prev.nnz()
        + local.sha_first.nnz()
        + last.nnz()
        + local.p_map.nnz()
        - 2 * canceled;
    let mut hash = blake3::Hasher::new();
    hash.update(b"bitz/sha256-ecdsa/map/v1");
    hash.update(&local.digest);
    hash.update(&last.digest());
    hash.update(&(n as u64).to_le_bytes());
    let h_layout = params(h_bits);
    let f_layout = params(f_bits);
    let mut map = Sha256EcdsaMap {
        local: local.clone(),
        last,
        n,
        h_offset,
        f_offset,
        aliases: [0; P_INPUT_ALIAS],
        rows: h_layout.cells(),
        cols: f_layout.cells(),
        nnz,
        digest: *hash.finalize().as_bytes(),
    };
    map.aliases = array::from_fn(|c| map.p_source(c));
    Ok(PreparedSha256Ecdsa {
        local,
        map,
        log_n: log_compressions,
        mode,
        lambda,
        h_layout,
        f_layout,
        ligerito: crate::ligerito_flock::LigeritoSelection::for_target(lambda as usize)
            .resolve(f_bits - 7, lambda as usize)
            .map_err(error)?,
    })
}
