//! Checked Ligerito-only policy. Enclosing protocols own their security target;
//! this module never reads process environment or configures another PCS.

use super::*;

/// Geometry and decoding bound, independent of the enclosing security budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LigeritoSelection {
    /// Historical rate-1/2 UDR with fold grinding.
    ValidatedUdr,
    CustomJohnson {
        log_inv_rate: usize,
        initial_k: usize,
    },
    CustomUdr {
        log_inv_rate: usize,
        initial_k: usize,
        fold_grinding: bool,
    },
    Embedded(ligerito::LigeritoProfile),
}

impl LigeritoSelection {
    pub const JOHNSON: Self = Self::CustomJohnson {
        log_inv_rate: 1,
        initial_k: 4,
    };
    pub const MATCHED_UDR: Self = Self::CustomUdr {
        log_inv_rate: 1,
        initial_k: 4,
        fold_grinding: true,
    };

    /// Existing higher-target standalone profiles keep their historical policy.
    /// Composed 100-bit protocols explicitly choose JOHNSON for their components.
    pub const fn for_target(target: usize) -> Self {
        if target == 100 {
            Self::JOHNSON
        } else {
            Self::ValidatedUdr
        }
    }

    pub fn parse(request: &str, target: usize) -> Result<Self, String> {
        let request = request.trim();
        match request {
            "johnson" => return Ok(Self::JOHNSON),
            "udr" | "legacy-udr" => return Ok(Self::ValidatedUdr),
            "slim" => return Ok(Self::Embedded(ligerito::LigeritoProfile::Slim)),
            "slim3" => return Ok(Self::Embedded(ligerito::LigeritoProfile::Slim3)),
            "fast" => return Ok(Self::Embedded(ligerito::LigeritoProfile::Fast)),
            "secure" => return Ok(Self::Embedded(ligerito::LigeritoProfile::Secure)),
            _ => {}
        }
        let parts: Vec<_> = request.split(':').collect();
        if !(3..=4).contains(&parts.len()) || !matches!(parts[0], "custom" | "udr" | "udrg") {
            return Err(format!(
                "unknown Ligerito profile {request:?}; use custom:1:4 or udrg:1:4"
            ));
        }
        let parse = |part: &str| {
            part.parse::<usize>()
                .map_err(|_| format!("invalid Ligerito profile {request:?}"))
        };
        let log_inv_rate = parse(parts[1])?;
        let initial_k = parse(parts[2])?;
        if parts.len() == 4 && parse(parts[3])? != target {
            return Err(format!(
                "Ligerito profile target conflicts with the enclosing {target}-bit component budget"
            ));
        }
        Ok(if parts[0] == "custom" {
            Self::CustomJohnson {
                log_inv_rate,
                initial_k,
            }
        } else {
            Self::CustomUdr {
                log_inv_rate,
                initial_k,
                fold_grinding: parts[0] == "udrg",
            }
        })
    }

    pub fn name(self) -> String {
        match self {
            Self::ValidatedUdr => "udrg:1:4".into(),
            Self::CustomJohnson {
                log_inv_rate,
                initial_k,
            } => format!("custom:{log_inv_rate}:{initial_k}"),
            Self::CustomUdr {
                log_inv_rate,
                initial_k,
                fold_grinding,
            } => {
                format!(
                    "{}:{log_inv_rate}:{initial_k}",
                    if fold_grinding { "udrg" } else { "udr" }
                )
            }
            Self::Embedded(profile) => format!("{profile:?}").to_lowercase(),
        }
    }

    pub fn resolve(self, packed_vars: usize, target: usize) -> Result<ResolvedLigerito, String> {
        let m = packed_vars
            .checked_add(LOG_PACKING)
            .ok_or("Ligerito dimension overflow")?;
        if !(20..=35).contains(&m) || !(64..=128).contains(&target) {
            return Err(format!(
                "unsupported Ligerito configuration: committed-bit exponent {m}, target {target}; require m=20..35 and target=64..128"
            ));
        }
        let mut security = match self {
            Self::ValidatedUdr => try_udr_config_impl(m, 1, 4, Some(target), true)?,
            Self::CustomJohnson {
                log_inv_rate,
                initial_k,
            } => {
                validate_geometry(packed_vars, log_inv_rate, initial_k)?;
                try_custom_johnson_config_bits(m, log_inv_rate, initial_k, Some(target))?
            }
            Self::CustomUdr {
                log_inv_rate,
                initial_k,
                fold_grinding,
            } => {
                validate_geometry(packed_vars, log_inv_rate, initial_k)?;
                try_udr_config_impl(m, log_inv_rate, initial_k, Some(target), fold_grinding)?
            }
            Self::Embedded(profile) => {
                let source = ligerito::embedded_security_config(m, profile)
                    .ok_or_else(|| format!("no embedded {profile:?} Ligerito profile for m={m}"))?;
                let config = LigeritoSecurityConfig::from_toml_str(source)?;
                if config.target_security_bits != target {
                    return Err(
                        "embedded Ligerito target conflicts with the enclosing component budget"
                            .into(),
                    );
                }
                config
            }
        };
        security.hash = "blake3".into();
        security.validate()?;
        let ood_bits = ood_round_bits(&security, packed_vars);
        let johnson = security
            .levels
            .iter()
            .any(|level| matches!(level.regime, SoundnessRegime::JohnsonOod));
        if johnson != ood_bits.is_some() || ood_bits.is_some_and(|bits| !bits.is_finite()) {
            return Err("Ligerito regime and Round-0 collision accounting disagree".into());
        }
        let (prover, verifier) = security.to_prover_verifier_configs()?;
        let digest =
            *blake3::hash(&bincode::serialize(&security).map_err(|error| error.to_string())?)
                .as_bytes();
        Ok(ResolvedLigerito {
            selection: self,
            security,
            prover,
            verifier,
            ood_bits,
            digest,
        })
    }
}

fn validate_geometry(packed_vars: usize, rate: usize, initial_k: usize) -> Result<(), String> {
    if !(1..=4).contains(&rate) || initial_k == 0 || initial_k >= packed_vars {
        return Err(
            "Ligerito requires inverse-rate exponent 1..4 and 1 <= initial_k < packed variables"
                .into(),
        );
    }
    Ok(())
}

/// One validated source of truth for commitment, opening, OOD and reporting.
#[derive(Clone, Debug)]
pub struct ResolvedLigerito {
    selection: LigeritoSelection,
    security: LigeritoSecurityConfig,
    prover: LigProverConfig,
    verifier: LigVerifierConfig,
    ood_bits: Option<f64>,
    digest: [u8; 32],
}

impl ResolvedLigerito {
    pub fn selection(&self) -> LigeritoSelection {
        self.selection
    }
    pub fn security(&self) -> &LigeritoSecurityConfig {
        &self.security
    }
    pub fn prover(&self) -> &LigProverConfig {
        &self.prover
    }
    pub fn verifier(&self) -> &LigVerifierConfig {
        &self.verifier
    }
    pub fn ood_bits(&self) -> Option<f64> {
        self.ood_bits
    }
    pub fn round0(&self, target: u32) -> Result<Option<OodRoundParams>, String> {
        self.ood_bits
            .map(|bits| {
                let grinding_bits = (target as f64 - bits).ceil().max(0.) as u32;
                if grinding_bits > crate::piop::spartan::profile::MAX_DERIVED_GRINDING_BITS {
                    return Err("Round 0 exceeds the 24-bit derived grinding cap".into());
                }
                Ok(OodRoundParams { grinding_bits })
            })
            .transpose()
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    /// Version the enclosing transcript and bind the actual resolved policy
    /// before any OOD, projection or PIOP challenge.
    pub fn bind(&self, transcript: &mut impl Transcript) {
        transcript.absorb_slice(b"bitz/ligerito-policy/early-ood/v1");
        transcript.absorb_slice(&self.digest);
    }
    /// Complete machine-readable identity. OOD work belongs to the enclosing
    /// protocol budget, which may exceed the native opener target.
    pub fn report(&self, requested: &str, ood: Option<OodRoundParams>) -> serde_json::Value {
        serde_json::json!({
            "requested_profile": requested.trim(), "resolved_profile": self.selection.name(),
            "regime": if self.ood_bits.is_some() { "johnson" } else { "udr" },
            "target_bits": self.security.target_security_bits,
            "protocol_version": "bitz/ligerito-policy/early-ood/v1",
            "configuration_fingerprint": self.digest.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "outer_ood": ood.is_some(), "outer_ood_grinding_bits": ood.map(|p| p.grinding_bits),
            "outer_ood_raw_bits": self.ood_bits,
            "recursive_ood": self.prover.ood_samples, "configuration": self.security,
        })
    }

    /// Hex framing keeps JSON string whitespace intact in key=value output.
    pub fn encode_report(report: &serde_json::Value) -> String {
        serde_json::to_vec(report)
            .expect("JSON value")
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    pub fn decode_report(encoded: &str) -> Result<serde_json::Value, String> {
        if !encoded.len().is_multiple_of(2) || !encoded.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err("invalid Ligerito identity encoding".into());
        }
        let bytes = (0..encoded.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&encoded[i..i + 2], 16).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let report = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        Self::validate_report(&report)?;
        Ok(report)
    }

    /// Reject missing, historical, or internally inconsistent result identities.
    /// Resolve again from the recorded selection instead of trusting a label.
    pub fn validate_report(report: &serde_json::Value) -> Result<(), String> {
        let config: LigeritoSecurityConfig =
            serde_json::from_value(report["configuration"].clone()).map_err(|e| e.to_string())?;
        let requested = report["requested_profile"]
            .as_str()
            .ok_or("missing requested profile")?;
        let resolved = LigeritoSelection::parse(requested, config.target_security_bits)?
            .resolve(config.log_n, config.target_security_bits)?;
        let ood = if report["outer_ood"] == true {
            let grinding_bits = report["outer_ood_grinding_bits"]
                .as_u64()
                .ok_or("missing OOD grinding")?;
            if grinding_bits > 24 {
                return Err("OOD grinding exceeds the derived cap".into());
            }
            Some(OodRoundParams {
                grinding_bits: grinding_bits as u32,
            })
        } else {
            None
        };
        if ood.is_some() != resolved.ood_bits().is_some()
            || resolved.report(requested, ood) != *report
        {
            return Err("inconsistent Ligerito result identity".into());
        }
        Ok(())
    }

    pub fn into_configs(self) -> ((LigProverConfig, LigVerifierConfig), Option<f64>) {
        ((self.prover, self.verifier), self.ood_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_profile_selection() {
        assert_eq!(
            LigeritoSelection::parse("custom:1:4", 100).unwrap(),
            LigeritoSelection::JOHNSON
        );
        assert_eq!(
            LigeritoSelection::parse("udrg:1:4:106", 106).unwrap(),
            LigeritoSelection::MATCHED_UDR
        );
        for invalid in [
            "",
            "typo",
            "custom:3",
            "custom:3:4:100:extra",
            "custom:x:4",
            "custom:3:4:99",
        ] {
            assert!(LigeritoSelection::parse(invalid, 100).is_err(), "{invalid}");
        }
    }

    #[test]
    fn production_geometry_and_ood_are_resolved_together() {
        for m in 20..=35 {
            for selection in [LigeritoSelection::JOHNSON, LigeritoSelection::MATCHED_UDR] {
                let resolved = selection.resolve(m - LOG_PACKING, 100).unwrap();
                assert_eq!(resolved.prover().initial_k, 4);
                assert_eq!(resolved.prover().log_inv_rates[0], 1);
                assert_eq!(resolved.prover().merkle_hash, HashKind::Blake3);
                assert_eq!(
                    resolved.ood_bits().is_some(),
                    selection == LigeritoSelection::JOHNSON
                );
                resolved.security().validate().unwrap();
            }
        }
        assert!(LigeritoSelection::JOHNSON.resolve(12, 100).is_err());
        assert!(LigeritoSelection::JOHNSON.resolve(29, 100).is_err());
        assert!(LigeritoSelection::JOHNSON.resolve(20, 128).is_err());
    }
}
