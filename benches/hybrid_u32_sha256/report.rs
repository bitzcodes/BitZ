//! CSV projections and configuration identities shared by the child and sweep.
use serde::Serialize;

/// Binius-Ligerito has one binary opener per oracle and a whole-protocol target.
/// It does not use the single-opener ResolvedLigerito policy of the other modes.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BiniusLigeritoIdentity {
    schema: String,
    target_bits: u32,
    component_bits: usize,
    log_inv_rate: usize,
    accounting: String,
    union_bound_bits: f64,
    round_by_round_bits: f64,
    oracles: Vec<BinaryOpenerIdentity>,
}

#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BinaryOpenerIdentity {
    packed_log: usize,
    configuration: serde_json::Value,
    ood_grinding_bits: u32,
}

impl BinaryOpenerIdentity {
    fn new(opener: &bitz::binary_pcs::BinaryPcs) -> Result<Self, super::AnyError> {
        Ok(Self {
            packed_log: opener.packed_log(),
            configuration: serde_json::to_value(opener.config())?,
            ood_grinding_bits: opener.ood_grinding_bits(),
        })
    }
}

impl BiniusLigeritoIdentity {
    const SCHEMA: &str = "bitz/binius-ligerito-pcs/v1";

    pub(super) fn new(prepared: &bitz::binius_ligerito::Prepared) -> Result<Self, super::AnyError> {
        Ok(Self {
            schema: Self::SCHEMA.into(),
            target_bits: prepared.security().target_bits,
            component_bits: prepared.component_bits(),
            log_inv_rate: prepared.log_inv_rate(),
            accounting: prepared.accounting().name().into(),
            union_bound_bits: prepared.security().union_bound_bits,
            round_by_round_bits: prepared.security().round_by_round_bits,
            oracles: (0..prepared.oracle_specs().len())
                .map(|i| BinaryOpenerIdentity::new(prepared.opener(i)))
                .collect::<Result<_, _>>()?,
        })
    }

    pub(super) fn validate(&self) -> Result<(), super::AnyError> {
        use bitz::binius_ligerito::{MAX_COMPONENT_BITS, MIN_COMPONENT_BITS, TARGET_BITS};
        if self.schema != Self::SCHEMA
            || self.target_bits != TARGET_BITS
            || !(MIN_COMPONENT_BITS..=MAX_COMPONENT_BITS).contains(&self.component_bits)
            || !(1..=3).contains(&self.log_inv_rate)
            || !self.union_bound_bits.is_finite()
            || !self.round_by_round_bits.is_finite()
            || self.oracles.is_empty()
        {
            return Err("invalid Binius-Ligerito identity or security budget".into());
        }
        let gated_bits = match self.accounting.as_str() {
            "union-bound" => self.union_bound_bits,
            "round-by-round" => self.round_by_round_bits,
            _ => return Err("invalid Binius-Ligerito accounting model".into()),
        };
        if gated_bits < f64::from(TARGET_BITS) {
            return Err("Binius-Ligerito security gate not met".into());
        }
        // Re-derive every ladder and Round-0 setting instead of accepting a
        // regime label or treating the whole-protocol target as an opener target.
        for oracle in &self.oracles {
            let opener = bitz::binary_pcs::BinaryPcs::with_log_inv_rate(oracle.packed_log, self.component_bits, self.log_inv_rate)?;
            if *oracle != BinaryOpenerIdentity::new(&opener)? {
                return Err("inconsistent Binius-Ligerito oracle configuration".into());
            }
        }
        Ok(())
    }
}

#[derive(Serialize)]
pub(super) struct HybridRow<'a> {
    pub mode: &'a str,
    pub iteration: usize,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub setup_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub witness_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub witness_commit_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub continuation_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub total_prover_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub verify_ms: f64,
    pub proof_bytes: usize,
    pub peak_rss_kib: u64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub piop_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub iop_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub mul_piop_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub sha_piop_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub mul_opening_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub joint_sumcheck_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub shared_opening_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub ood_round_ms: f64,
}
impl HybridRow<'_> {
    pub const HEADER: [&'static str; 18] = [
        "mode",
        "iteration",
        "setup_ms",
        "witness_ms",
        "witness_commit_ms",
        "continuation_ms",
        "total_prover_ms",
        "verify_ms",
        "proof_bytes",
        "peak_rss_kib",
        "piop_ms",
        "iop_ms",
        "mul_piop_ms",
        "sha_piop_ms",
        "mul_opening_ms",
        "joint_sumcheck_ms",
        "shared_opening_ms",
        "ood_round_ms",
    ];
}

#[derive(Serialize)]
pub(super) struct NativeRow<'a> {
    pub mode: &'a str,
    pub iteration: usize,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub setup_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub witness_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub total_prover_ms: f64,
    #[serde(serialize_with = "super::output::csv_format::three_decimals")]
    pub verify_ms: f64,
    pub proof_bytes: usize,
    pub peak_rss_kib: u64,
}
impl NativeRow<'_> {
    pub fn header(mode: &str) -> [&'static str; 8] {
        let mut header = [
            "mode",
            "iteration",
            "setup_ms",
            "witness_ms",
            "total_prover_ms",
            "verify_ms",
            "proof_bytes",
            "peak_rss_kib",
        ];
        if mode == "separate" {
            header[6] = "proof_payload_bytes_estimate";
        }
        header
    }
}

#[derive(Default, Serialize, serde::Deserialize)]
#[serde(default)]
pub(super) struct SummaryRow {
    pub mode: String,
    pub multiplication_relation: String,
    pub mul_log: u32,
    pub sha_log: u32,
    pub multiplications: usize,
    pub sha_compressions: usize,
    // Retain child numeric lexemes (including decimal precision); absent metrics stay empty.
    pub iteration: String,
    pub setup_ms: String,
    pub witness_ms: String,
    pub witness_commit_ms: String,
    pub continuation_ms: String,
    pub total_prover_ms: String,
    pub verify_ms: String,
    pub proof_bytes: String,
    pub proof_payload_bytes_estimate: String,
    pub peak_rss_kib: String,
    pub piop_ms: String,
    pub iop_ms: String,
    pub mul_piop_ms: String,
    pub sha_piop_ms: String,
    pub mul_opening_ms: String,
    pub joint_sumcheck_ms: String,
    pub shared_opening_ms: String,
    pub ood_round_ms: String,
    pub ligerito_hex: String,
}
impl SummaryRow {
    pub const HEADER: [&'static str; 25] = [
        "mode",
        "multiplication_relation",
        "mul_log",
        "sha_log",
        "multiplications",
        "sha_compressions",
        "iteration",
        "setup_ms",
        "witness_ms",
        "witness_commit_ms",
        "continuation_ms",
        "total_prover_ms",
        "verify_ms",
        "proof_bytes",
        "proof_payload_bytes_estimate",
        "peak_rss_kib",
        "piop_ms",
        "iop_ms",
        "mul_piop_ms",
        "sha_piop_ms",
        "mul_opening_ms",
        "joint_sumcheck_ms",
        "shared_opening_ms",
        "ood_round_ms",
        "ligerito_hex",
    ];
}

#[cfg(test)]
mod reporting_tests {
    use super::super::output;
    use super::*;

    #[test]
    fn csv_contract_hybrid_and_native_modes() {
        let mut csv = output::csv_writer(Vec::new());
        csv.write_record(HybridRow::HEADER).unwrap();
        csv.flush().unwrap();
        let header = "mode,iteration,setup_ms,witness_ms,witness_commit_ms,continuation_ms,total_prover_ms,verify_ms,proof_bytes,peak_rss_kib,piop_ms,iop_ms,mul_piop_ms,sha_piop_ms,mul_opening_ms,joint_sumcheck_ms,shared_opening_ms,ood_round_ms\n";
        assert_eq!(csv.get_ref(), header.as_bytes());
        csv.serialize(HybridRow {
            mode: "hybrid",
            iteration: 0,
            setup_ms: 1.2346,
            witness_ms: 2.0,
            witness_commit_ms: 3.0,
            continuation_ms: 4.0,
            total_prover_ms: 7.0,
            verify_ms: 5.0,
            proof_bytes: 1024,
            peak_rss_kib: 0,
            piop_ms: 6.0,
            iop_ms: 7.0,
            mul_piop_ms: 8.0,
            sha_piop_ms: 9.0,
            mul_opening_ms: 10.0,
            joint_sumcheck_ms: 11.0,
            shared_opening_ms: 12.0,
            ood_round_ms: 13.0,
        })
        .unwrap();
        assert_eq!(
            String::from_utf8(csv.into_inner().unwrap()).unwrap(),
            format!(
                "{header}hybrid,0,1.235,2.000,3.000,4.000,7.000,5.000,1024,0,6.000,7.000,8.000,9.000,10.000,11.000,12.000,13.000\n"
            )
        );
        for mode in ["all-binius", "binius-ligerito", "separate"] {
            let mut csv = output::csv_writer(Vec::new());
            csv.write_record(NativeRow::header(mode)).unwrap();
            let size = if mode == "separate" {
                "proof_payload_bytes_estimate"
            } else {
                "proof_bytes"
            };
            let header = format!(
                "mode,iteration,setup_ms,witness_ms,total_prover_ms,verify_ms,{size},peak_rss_kib\n"
            );
            csv.flush().unwrap();
            assert_eq!(csv.get_ref(), header.as_bytes());
            csv.serialize(NativeRow {
                mode,
                iteration: 0,
                setup_ms: 1.2346,
                witness_ms: 2.0,
                total_prover_ms: 3.0,
                verify_ms: 4.0,
                proof_bytes: 1024,
                peak_rss_kib: 0,
            })
            .unwrap();
            assert_eq!(
                String::from_utf8(csv.into_inner().unwrap()).unwrap(),
                format!("{header}{mode},0,1.235,2.000,3.000,4.000,1024,0\n")
            );
        }
    }
}
