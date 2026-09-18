//! Grinding guards for challenge blocks in the pinned Flock schedule (Johnson+OOD or UDR).
//!
//! A native PoW starts a block. Otherwise the first draw after an observation
//! starts one. Consecutive draws (including query retries and alpha) share it.
use super::*;

/// Consecutive challenge draws covered by one grinding requirement.
#[derive(Clone, Debug)]
pub(crate) struct ChallengeBlock {
    pub label: String,
    pub native_bits: Option<u32>,
    pub raw_error: f64,
    pub bits: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct GrindingPlan {
    pub blocks: Vec<ChallengeBlock>,
    final_verifier_draws: usize,
}

impl GrindingPlan {
    pub fn resolve(config: &ligerito::LigeritoSecurityConfig, target: u32) -> Result<Self, String> {
        config.validate()?;
        let mut blocks: Vec<ChallengeBlock> = Vec::new();
        let gf_error = 2f64.powi(-128);
        for (level, params) in config.levels.iter().enumerate() {
            let (pg, query) = params.paper_predicted_bits();
            let folds = if level == 0 {
                config.initial_k
            } else {
                params.k_recursive
            };
            for round in 0..folds {
                let native = (params.fold_grinding_bits as u32).saturating_sub(round as u32);
                let error = 2f64.powf(-pg - round as f64) + 2. * gf_error;
                if level > 0 && round == 0 && native == 0 {
                    // Introduce beta and the unground first fold have no
                    // intervening observation. Their errors share one budget.
                    let last = blocks.last_mut().ok_or("missing introduce block")?;
                    last.raw_error += error;
                    last.label.push_str("+fold0");
                } else {
                    blocks.push(ChallengeBlock {
                        label: format!("fold/{level}/{round}"),
                        native_bits: (native > 0).then_some(native),
                        raw_error: error,
                        bits: 0,
                    });
                }
            }
            // The next root is observed after these folds, before this level's
            // queries. Each OOD evaluation/introduction is an observation. Its
            // beta and the following sample's coordinates have no observation
            // between them and therefore share one uninterrupted block.
            if let Some(next) = config.levels.get(level + 1) {
                if next.ood_samples > 0 {
                    let mut single = next.clone();
                    single.ood_samples = 1;
                    let collision = 2f64.powf(-single.paper_predicted_ood_bits().ok_or("OOD without a Johnson bound")?);
                    for sample in 0..next.ood_samples {
                        if sample == 0 {
                            blocks.push(ChallengeBlock {
                                label: format!("ood/{}/{sample}", level + 1),
                                native_bits: None,
                                raw_error: collision,
                                bits: 0,
                            });
                        } else {
                            let block = blocks.last_mut().ok_or("missing OOD beta block")?;
                            block.raw_error += collision;
                            block.label.push_str(&format!("+ood/{}/{sample}", level + 1));
                        }
                        blocks.push(ChallengeBlock {
                            label: format!("ood-beta/{}/{sample}", level + 1),
                            native_bits: None,
                            raw_error: gf_error,
                            bits: 0,
                        });
                    }
                }
            }
            let alpha_vars = params.queries.next_power_of_two().ilog2();
            blocks.push(ChallengeBlock {
                label: format!("queries/{level}"),
                native_bits: Some(params.grinding_bits as u32),
                // Count alpha even at the final level where the prover omits
                // the verifier's final alpha and beta draws. It does not split the block.
                raw_error: 2f64.powf(-query)
                    + (f64::from(alpha_vars)
                        + if level + 1 == config.levels.len() {
                            1.
                        } else {
                            0.
                        })
                        * gf_error,
                bits: 0,
            });
            if level + 1 < config.levels.len() {
                blocks.push(ChallengeBlock {
                    label: format!("introduce/{level}"),
                    native_bits: None,
                    raw_error: gf_error,
                    bits: 0,
                });
            }
        }
        for block in &mut blocks {
            block.bits = (f64::from(target) + block.raw_error.log2()).ceil().max(0.) as u32;
            block.bits = block.bits.max(block.native_bits.unwrap_or(0));
            if block.bits > 32 {
                return Err(format!("{} exceeds the 32-bit grinding cap", block.label));
            }
        }
        let final_verifier_draws = config
            .levels
            .last()
            .ok_or("empty Flock schedule")?
            .queries
            .next_power_of_two()
            .ilog2() as usize
            + 1;
        Ok(Self {
            blocks,
            final_verifier_draws,
        })
    }
}

pub(crate) enum GrindingNonces<'a> {
    Prove(&'a mut Vec<u64>),
    Verify { values: &'a [u64], cursor: usize },
}

pub(crate) struct GrindingContext<'a> {
    pub plan: &'a GrindingPlan,
    pub nonces: GrindingNonces<'a>,
}

pub(crate) struct GrindingChallenger<'a, 'b, T: Transcript + Send> {
    inner: ZincChallenger<'a, T>,
    security: &'a mut GrindingContext<'b>,
    next: usize,
    active: bool,
    valid: bool,
}

impl<'a, 'b, T: Transcript + Send> GrindingChallenger<'a, 'b, T> {
    pub fn new(transcript: &'a mut T, security: &'a mut GrindingContext<'b>) -> Self {
        Self {
            inner: ZincChallenger(transcript),
            security,
            next: 0,
            active: false,
            valid: true,
        }
    }
    fn begin(&mut self, native: Option<u32>) -> u32 {
        let Some(block) = self.security.plan.blocks.get(self.next) else {
            self.valid = false;
            return 0;
        };
        self.valid &= block.native_bits == native;
        // This protocol domain separator stays fixed across Rust type renames.
        self.inner.observe_label(b"bitz/flock/atomic/v1");
        self.inner.observe_bytes(&(self.next as u64).to_le_bytes());
        self.inner.observe_label(block.label.as_bytes());
        self.inner.observe_bytes(&block.bits.to_le_bytes());
        self.inner.observe_bytes(&[u8::from(native.is_some())]);
        self.next += 1;
        self.active = true;
        block.bits
    }
    pub fn finish(mut self) -> bool {
        // The pinned prover omits alpha and beta for the final verifier-only check.
        // Consume that suffix so callers can safely continue the transcript.
        if matches!(self.security.nonces, GrindingNonces::Prove(_)) {
            for _ in 0..self.security.plan.final_verifier_draws {
                self.inner.sample_f128();
            }
        }
        self.valid
            && self.next == self.security.plan.blocks.len()
            && match &self.security.nonces {
                GrindingNonces::Prove(_) => true,
                GrindingNonces::Verify { values, cursor } => values.len() == *cursor,
            }
    }
}

impl<T: Transcript + Send> Challenger for GrindingChallenger<'_, '_, T> {
    fn observe_label(&mut self, label: &[u8]) {
        self.active = false;
        self.inner.observe_label(label);
    }
    fn observe_f128(&mut self, value: Gf128) {
        self.active = false;
        self.inner.observe_f128(value);
    }
    fn observe_f128_slice(&mut self, values: &[Gf128]) {
        self.active = false;
        self.inner.observe_f128_slice(values);
    }
    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.active = false;
        self.inner.observe_bytes(bytes);
    }
    fn sample_f128(&mut self) -> Gf128 {
        if !self.active {
            let bits = self.begin(None);
            if bits > 0 {
                match &mut self.security.nonces {
                    GrindingNonces::Prove(nonces) => nonces.push(self.inner.grind_pow(bits)),
                    GrindingNonces::Verify { values, cursor } => {
                        let nonce = values.get(*cursor).copied().unwrap_or(0);
                        self.valid &= *cursor < values.len() && self.inner.verify_pow(nonce, bits);
                        *cursor += 1;
                    }
                }
            }
        }
        self.inner.sample_f128()
    }
    fn grind_pow(&mut self, bits: u32) -> u64 {
        let bits = self.begin(Some(bits));
        self.inner.grind_pow(bits)
    }
    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        let bits = self.begin(Some(bits));
        let valid = self.inner.verify_pow(nonce, bits);
        self.valid &= valid;
        valid && self.valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Blake3Transcript;

    #[test]
    fn vectors_and_native_pow_share_blocks_and_replay_exactly() {
        let plan = GrindingPlan {
            blocks: vec![
                ChallengeBlock {
                    label: "host".into(),
                    native_bits: None,
                    raw_error: 0.01,
                    bits: 3,
                },
                ChallengeBlock {
                    label: "queries".into(),
                    native_bits: Some(4),
                    raw_error: 0.01,
                    bits: 6,
                },
            ],
            final_verifier_draws: 0,
        };
        let mut pt = Blake3Transcript::new();
        let mut vt = Blake3Transcript::new();
        let mut nonces = Vec::new();
        let mut security = GrindingContext {
            plan: &plan,
            nonces: GrindingNonces::Prove(&mut nonces),
        };
        let mut p = GrindingChallenger::new(&mut pt, &mut security);
        let host: Vec<_> = (0..7).map(|_| p.sample_f128()).collect();
        // No observation between the host vector and native grind. The native
        // grind still starts a new block, upgrading its existing nonce to 6 bits.
        let native = p.grind_pow(4);
        let queries: Vec<_> = (0..40).map(|_| p.sample_f128()).collect();
        assert!(p.finish());
        assert_eq!(nonces.len(), 1, "no per-coordinate or per-query host nonce");
        let mut security = GrindingContext {
            plan: &plan,
            nonces: GrindingNonces::Verify {
                values: &nonces,
                cursor: 0,
            },
        };
        let mut v = GrindingChallenger::new(&mut vt, &mut security);
        for x in host {
            assert_eq!(v.sample_f128(), x);
        }
        assert!(v.verify_pow(native, 4));
        for x in queries {
            assert_eq!(v.sample_f128(), x);
        }
        assert!(v.finish());
        assert_eq!(pt.get_challenge::<u128>(), vt.get_challenge::<u128>());
    }
}
