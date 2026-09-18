use super::{Result, Sha256EcdsaProof, error};
use crate::piop::spartan::SpartanField as _;
use crate::{
    ligerito_flock::IntEvalRsLigVirtProof,
    piop::spartan::{
        SpartanField,
        bitz::SpartanBitzField as F,
        sumcheck::{OuterSumcheckProof, SumcheckProof},
    },
    proof_codec::{Reader, Writer},
};
use field::Uint;

const MAGIC: &[u8] = b"BITZSE03";

impl Sha256EcdsaProof {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.bytes(MAGIC);
        let field = field::FpCtx::from_prime_u128(self.modulus);
        w.u128(self.modulus);
        w.bytes(&self.initial_nonce.to_le_bytes());
        w.bytes(&self.batch_nonce.to_le_bytes());
        write_nonces(&mut w, &self.flock_nonces);
        write_rounds(&mut w, &self.outer.sumcheck, &field);
        for x in [
            &self.outer.az_mle_claim,
            &self.outer.bz_mle_claim,
            &self.outer.cz_mle_claim,
        ] {
            w.u128(u128::from(field.to_integer(x)));
        }
        write_nonces(&mut w, &self.outer_nonces);
        write_rounds(&mut w, &self.inner, &field);
        write_nonces(&mut w, &self.inner_nonces);
        let opening = self.opening.to_bytes();
        w.len(opening.len());
        w.bytes(&opening);
        w.into_vec()
    }

    /// The encoded modulus is only a decoding context. Verification independently
    /// re-derives it from the bound commitment, relation and public statement.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        if r.take(MAGIC.len()).map_err(error)? != MAGIC {
            return Err(error("invalid proof version"));
        }
        let q = r.u128().map_err(error)?;
        if !((1u128 << 112)..(1u128 << 113)).contains(&q) {
            return Err(error("invalid encoded modulus"));
        }
        let cfg = F::make_cfg(&Uint::from(q)).map_err(|_| error("invalid encoded field"))?;
        F::validate_config(&cfg).map_err(error)?;
        let initial_nonce = u64::from_le_bytes(r.take(8).map_err(error)?.try_into().unwrap());
        let batch_nonce = u64::from_le_bytes(r.take(8).map_err(error)?.try_into().unwrap());
        let flock_nonces = read_nonces(&mut r)?;
        let sumcheck = read_rounds(&mut r, q, &cfg)?;
        let az_mle_claim = read_field(&mut r, q, &cfg)?;
        let bz_mle_claim = read_field(&mut r, q, &cfg)?;
        let cz_mle_claim = read_field(&mut r, q, &cfg)?;
        let outer_nonces = read_nonces(&mut r)?;
        let inner = read_rounds(&mut r, q, &cfg)?;
        let inner_nonces = read_nonces(&mut r)?;
        let len = r.len().map_err(error)?;
        let opening =
            IntEvalRsLigVirtProof::from_bytes(r.take(len).map_err(error)?).map_err(error)?;
        if r.remaining() != 0 {
            return Err(error("trailing proof bytes"));
        }
        Ok(Self {
            modulus: q,
            initial_nonce,
            batch_nonce,
            flock_nonces,
            outer: OuterSumcheckProof {
                sumcheck,
                az_mle_claim,
                bz_mle_claim,
                cz_mle_claim,
            },
            outer_nonces,
            inner,
            inner_nonces,
            opening,
        })
    }
}

fn write_rounds<const C: usize>(
    w: &mut Writer,
    proof: &SumcheckProof<F, C>,
    field: &field::FpCtx<2>,
) {
    w.len(proof.round_polynomials.len());
    for round in &proof.round_polynomials {
        for x in round {
            w.u128(u128::from(field.to_integer(x)));
        }
    }
}
fn read_field(
    r: &mut Reader<'_>,
    q: u128,
    cfg: &<F as crate::piop::spartan::SpartanField>::Config,
) -> Result<F> {
    let x = r.u128().map_err(error)?;
    if x >= q {
        return Err(error("noncanonical field element"));
    }
    Ok(F::from_with_cfg(x, cfg))
}
fn read_rounds<const C: usize>(
    r: &mut Reader<'_>,
    q: u128,
    cfg: &<F as crate::piop::spartan::SpartanField>::Config,
) -> Result<SumcheckProof<F, C>> {
    let len = r.len().map_err(error)?;
    if len > 31 || len > r.remaining() / (16 * C) {
        return Err(error("invalid sumcheck length"));
    }
    let mut round_polynomials = Vec::with_capacity(len);
    for _ in 0..len {
        let mut round = std::array::from_fn(|_| F::zero_with_cfg(cfg));
        for x in &mut round {
            *x = read_field(r, q, cfg)?;
        }
        round_polynomials.push(round);
    }
    Ok(SumcheckProof { round_polynomials })
}
fn write_nonces(w: &mut Writer, nonces: &[u64]) {
    w.len(nonces.len());
    for nonce in nonces {
        w.bytes(&nonce.to_le_bytes());
    }
}
fn read_nonces(r: &mut Reader<'_>) -> Result<Vec<u64>> {
    let len = r.len().map_err(error)?;
    if len > 31 || len > r.remaining() / 8 {
        return Err(error("invalid nonce count"));
    }
    (0..len)
        .map(|_| {
            Ok(u64::from_le_bytes(
                r.take(8).map_err(error)?.try_into().unwrap(),
            ))
        })
        .collect()
}
