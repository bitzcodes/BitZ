//! Versioned, bounded proof encoding. The runtime prime is derived from the
//! public statement while decoding; field encodings must be canonical for it.
use super::{Error, HybridProof, PreparedHybrid, Statement, mul, opening, sumcheck};
use crate::{
    ligerito::RingSwitchProof,
    ligerito_flock::OodRound,
    merged_forest::{MergedForestProof, MergedLayer},
    piop::spartan::{
        SpartanField,
        bitz::SpartanBitzField as Q,
        sumcheck::{OuterSumcheckProof, SumcheckProof as QSumcheck},
        univariate_skip::{
            UnivariateSkipOuterSumcheckProof, UnivariateSkipProof, UnivariateSkipSpartanPiopProof,
        },
    },
    proof_codec::{CodecError, Reader, Writer},
};
use bincode::Options;

use flock_core::field::Gf128;

const MAGIC: &[u8; 8] = b"BZSH\x06\0\0\0";
const MAX_PROOF_BYTES: usize = 64 << 20;

fn count(r: &mut Reader<'_>, max: usize) -> Result<usize, CodecError> {
    let n = r.len()?;
    if n > max {
        Err(CodecError::NonCanonical)
    } else {
        Ok(n)
    }
}

fn write_q(w: &mut Writer, q: &Q, cfg: &field::FpCtx<2>) {
    w.bytes(&q.canonical_element_encoding(cfg));
}
fn read_q(
    r: &mut Reader<'_>,
    modulus: u128,
    cfg: &<Q as SpartanField>::Config,
) -> Result<Q, CodecError> {
    let x = r.u128()?;
    if x >= modulus {
        return Err(CodecError::NonCanonical);
    }
    Ok(Q::from_with_cfg(x, cfg))
}
fn write_rounds<const N: usize>(w: &mut Writer, rounds: &QSumcheck<Q, N>, cfg: &field::FpCtx<2>) {
    w.len(rounds.round_polynomials.len());
    for row in &rounds.round_polynomials {
        for q in row {
            write_q(w, q, cfg);
        }
    }
}
fn read_rounds<const N: usize>(
    r: &mut Reader<'_>,
    modulus: u128,
    cfg: &<Q as SpartanField>::Config,
) -> Result<QSumcheck<Q, N>, CodecError> {
    let n = count(r, 64)?;
    let mut round_polynomials = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = Vec::with_capacity(N);
        for _ in 0..N {
            row.push(read_q(r, modulus, cfg)?);
        }
        round_polynomials.push(row.try_into().map_err(|_| CodecError::NonCanonical)?);
    }
    Ok(QSumcheck { round_polynomials })
}

fn write_sc(w: &mut Writer, sc: &crate::piop::sumcheck::SumcheckProof<super::Gf>) {
    w.gf(&sc.claimed_sum);
    w.len(sc.messages.len());
    for message in &sc.messages {
        w.len(message.0.tail_evaluations.len());
        for x in &message.0.tail_evaluations {
            w.gf(x);
        }
    }
}
fn read_sc(
    r: &mut Reader<'_>,
) -> Result<crate::piop::sumcheck::SumcheckProof<super::Gf>, CodecError> {
    use crate::piop::sumcheck::prover::{NatEvaluatedPolyWithoutConstant, ProverMsg};
    let claimed_sum = r.gf()?;
    let n = count(r, 64)?;
    let mut messages = Vec::with_capacity(n);
    for _ in 0..n {
        let k = count(r, 3)?;
        let mut tail_evaluations = Vec::with_capacity(k);
        for _ in 0..k {
            tail_evaluations.push(r.gf()?);
        }
        messages.push(ProverMsg(NatEvaluatedPolyWithoutConstant {
            tail_evaluations,
        }));
    }
    Ok(crate::piop::sumcheck::SumcheckProof {
        claimed_sum,
        messages,
    })
}

fn write_f(w: &mut Writer, f: Gf128) {
    w.gf(&(f));
}
fn read_f(r: &mut Reader<'_>) -> Result<Gf128, CodecError> {
    r.gf()
}

impl HybridProof {
    /// Canonical wire representation, including both initial Merkle multiproofs.
    /// Round 0 comes first, in transcript order: its value and (when the
    /// round grinds) its nonce precede the multiplication prefix, whose
    /// transcript-derived prime depends on them.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.bytes(MAGIC);
        w.len(usize::from(self.opening.ood.is_some()));
        if let Some(ood) = &self.opening.ood {
            w.gf(&ood.y);
            w.len(usize::from(ood.nonce.is_some()));
            if let Some(nonce) = ood.nonce {
                w.bytes(&nonce.to_le_bytes());
            }
        }
        let p = &self.multiplication;
        w.bytes(&p.initial_nonce.to_le_bytes());
        w.bytes(&p.terminal_nonce.to_le_bytes());
        w.len(p.piop_nonces.len());
        for nonce in &p.piop_nonces {
            w.bytes(&nonce.to_le_bytes());
        }
        let s = &p.spartan;
        w.len(s.outer.skip.skip_vars as usize);
        w.len(s.outer.skip.finite_q_evaluations.len());
        for q in &s.outer.skip.finite_q_evaluations {
            write_q(&mut w, q, &p.field);
        }
        write_q(&mut w, &s.outer.skip.q_at_infinity, &p.field);
        write_rounds(&mut w, &s.outer.tail.sumcheck, &p.field);
        for q in [
            &s.outer.tail.az_mle_claim,
            &s.outer.tail.bz_mle_claim,
            &s.outer.tail.cz_mle_claim,
        ] {
            write_q(&mut w, q, &p.field);
        }
        write_rounds(&mut w, &s.inner, &p.field);
        w.len(p.sums.len());
        for &sum in &p.sums {
            w.u128(sum);
        }
        w.len(p.forest.layers.len());
        for layer in &p.forest.layers {
            w.len(usize::from(layer.sc_x.is_some()));
            if let Some(sc) = &layer.sc_x {
                write_sc(&mut w, sc);
            }
            write_sc(&mut w, &layer.sc_c);
            w.gf(&layer.pair.0);
            w.gf(&layer.pair.1);
            // This protocol always uses the binary forest, regardless of
            // environment variables controlling standalone BitZ schedules.
            assert!(layer.pair2.is_none());
        }
        w.len(self.sha.len());
        for &word in &self.sha {
            w.u128(word);
        }
        w.len(self.joint.rounds.len());
        for row in &self.joint.rounds {
            for &f in row {
                write_f(&mut w, f);
            }
        }
        write_f(&mut w, self.joint.value);
        w.len(self.opening.ring.s_v.len());
        for f in &self.opening.ring.s_v {
            w.gf(f);
        }
        let bytes =
            bincode::serialize(&self.opening.ligerito).expect("Ligerito proof serialization");
        w.len(bytes.len());
        w.bytes(&bytes);
        for path in &self.opening.paths {
            w.len(path.len());
            for hash in path {
                w.bytes(hash);
            }
        }
        w.into_vec()
    }
}

impl PreparedHybrid {
    /// Decode for this prepared relation and public statement. Verification
    /// remains a separate operation; successful decoding does not accept a proof.
    pub fn proof_from_bytes(
        &self,
        statement: &Statement,
        bytes: &[u8],
    ) -> Result<HybridProof, Error> {
        if bytes.len() > MAX_PROOF_BYTES {
            return Err(CodecError::NonCanonical.into());
        }
        let mut r = Reader::new(bytes);
        if r.take(8)? != MAGIC {
            return Err(CodecError::NonCanonical.into());
        }
        let has_ood = count(&mut r, 1)? == 1;
        if has_ood != self.ood.is_some() {
            return Err(CodecError::NonCanonical.into());
        }
        let ood = if has_ood {
            let y = r.gf()?;
            let has_nonce = count(&mut r, 1)? == 1;
            if has_nonce != self.ood.is_some_and(|params| params.grinding_bits > 0) {
                return Err(CodecError::NonCanonical.into());
            }
            Some(OodRound {
                y,
                nonce: if has_nonce {
                    Some(u64::from_le_bytes(
                        r.take(8)?.try_into().expect("eight bytes"),
                    ))
                } else {
                    None
                },
            })
        } else {
            None
        };
        let initial_nonce = u64::from_le_bytes(r.take(8)?.try_into().expect("eight bytes"));
        let terminal_nonce = u64::from_le_bytes(r.take(8)?.try_into().expect("eight bytes"));
        let n = count(&mut r, 256)?;
        let mut piop_nonces = Vec::with_capacity(n);
        for _ in 0..n {
            piop_nonces.push(u64::from_le_bytes(
                r.take(8)?.try_into().expect("eight bytes"),
            ));
        }
        let (mut t, digest) = self.transcript(statement)?;
        // Replay Round 0: the prime below is derived from the transcript
        // state after it.
        opening::verify_ood(&mut t, &self.geometry, self.ood, ood.as_ref())?;
        let (q, cfg) = mul::decoding_config(&mut t, &self.multiplication, &digest, initial_nonce)?;
        let skip_vars = count(&mut r, 3)? as u8;
        if skip_vars != 3 {
            return Err(CodecError::NonCanonical.into());
        }
        let n = count(&mut r, 6)?;
        if n != 6 {
            return Err(CodecError::NonCanonical.into());
        }
        let mut finite = Vec::with_capacity(n);
        for _ in 0..n {
            finite.push(read_q(&mut r, q, &cfg)?);
        }
        let skip = UnivariateSkipProof {
            skip_vars,
            finite_q_evaluations: finite.into_boxed_slice(),
            q_at_infinity: read_q(&mut r, q, &cfg)?,
        };
        let tail = OuterSumcheckProof {
            sumcheck: read_rounds(&mut r, q, &cfg)?,
            az_mle_claim: read_q(&mut r, q, &cfg)?,
            bz_mle_claim: read_q(&mut r, q, &cfg)?,
            cz_mle_claim: read_q(&mut r, q, &cfg)?,
        };
        let spartan = UnivariateSkipSpartanPiopProof {
            outer: UnivariateSkipOuterSumcheckProof { skip, tail },
            inner: read_rounds(&mut r, q, &cfg)?,
        };
        let n = count(&mut r, self.multiplication.params().cols())?;
        let mut sums = Vec::with_capacity(n);
        for _ in 0..n {
            sums.push(r.u128()?);
        }
        let n = count(&mut r, 64)?;
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            let sc_x = if count(&mut r, 1)? == 1 {
                Some(read_sc(&mut r)?)
            } else {
                None
            };
            layers.push(MergedLayer {
                sc_x,
                sc_c: read_sc(&mut r)?,
                pair: (r.gf()?, r.gf()?),
                pair2: None,
            });
        }
        let n = count(&mut r, 1 << 16)?;
        let mut sha = Vec::with_capacity(n);
        for _ in 0..n {
            sha.push(r.u128()?);
        }
        let n = count(&mut r, self.geometry.bit_log())?;
        let mut rounds = Vec::with_capacity(n);
        for _ in 0..n {
            rounds.push([read_f(&mut r)?, read_f(&mut r)?]);
        }
        let joint = sumcheck::Proof {
            rounds,
            value: read_f(&mut r)?,
        };
        let n = count(&mut r, 128)?;
        let mut s_v = Vec::with_capacity(n);
        for _ in 0..n {
            s_v.push(r.gf()?);
        }
        let n = count(&mut r, MAX_PROOF_BYTES)?;
        let ligerito = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(n as u64)
            .reject_trailing_bytes()
            .deserialize(r.take(n)?)
            .map_err(|e| CodecError::Bincode(e.to_string()))?;
        let mut paths = [Vec::new(), Vec::new()];
        for path in &mut paths {
            let n = count(&mut r, 1 << 16)?;
            for _ in 0..n {
                path.push(r.take(32)?.try_into().expect("32 bytes"));
            }
        }
        if r.remaining() != 0 {
            return Err(CodecError::NonCanonical.into());
        }
        Ok(HybridProof {
            multiplication: mul::PrefixProof {
                field: cfg,
                initial_nonce,
                terminal_nonce,
                piop_nonces,
                spartan,
                sums,
                forest: MergedForestProof { layers },
            },
            sha,
            joint,
            opening: opening::Proof {
                ood,
                ring: RingSwitchProof { s_v },
                ligerito,
                paths,
            },
        })
    }
}
