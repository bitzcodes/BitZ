//! Print resolved Binius configurations without generating proofs.
//! Run with `--features binius64-bench --example binius_config_audit -- 7`.
use binius_circuits::sha256_ecdsa::Sha256Ecdsa;
use binius_core::constraint_system::ConstraintSystem;
use binius_frontend::CircuitBuilder;
use binius_hash::sha256::Sha256HashSuite;
use binius_verifier::Verifier;
use serde_json::{Value, json};

fn main() {
    let exponent: u8 = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "7".into())
        .parse()
        .unwrap();
    let builder = CircuitBuilder::new();
    Sha256Ecdsa::new(&builder, exponent).unwrap();
    let circuit = builder.build();
    let sha = audit(circuit.constraint_system());
    let builder = CircuitBuilder::new();
    let mask = builder.add_constant_64(u64::from(u32::MAX));
    for _ in 0..1 << 15 {
        let x = builder.add_witness();
        let y = builder.add_witness();
        let z = builder.add_witness();
        let (_, lo) = builder.imul(x, y);
        builder.assert_zero("x is u32", builder.shr(x, 32));
        builder.assert_zero("y is u32", builder.shr(y, 32));
        builder.assert_eq("product modulo 2^32", builder.band(lo, mask), z);
    }
    let circuit = builder.build();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "binius-config-audit/v1", "proofs_generated": false,
            "sha256_ecdsa": {"log_compressions":exponent,
                "message_bytes":binius_circuits::sha256_ecdsa::message_len(exponent).unwrap(),
                "resolved":sha},
            "u32_mod32": {"log_multiplications":15, "resolved":audit(circuit.constraint_system())},
        }))
        .unwrap()
    );
}

fn audit(cs: &ConstraintSystem) -> Value {
    let mut basefold = Vec::new();
    for log_inv_rate in [1, 3] {
        for target in [100, 128] {
            let verifier = Verifier::<Sha256HashSuite>::setup_with_security_bits(
                cs.clone(),
                log_inv_rate,
                target,
            )
            .unwrap();
            let fri = verifier.fri_params();
            basefold.push(json!({
                "log_inv_rate": fri.rs_code().log_inv_rate(),
                "query_target_bits": target,
                "queries": fri.n_test_queries(),
                "fold_arities": fri.fold_arities(),
                "log_message_len": fri.log_msg_len(),
                "security_scope": "FRI query phase only",
                "proximity_regime": "unique decoding",
            }));
        }
    }
    let ligerito = match bitz::binius_ligerito::Prepared::new(cs) {
        Ok(prepared) => ligerito_report(&prepared),
        Err(error) => json!({"supported":false, "error":error.to_string()}),
    };
    json!({
        "word_constraints": {"and":cs.n_and_constraints(), "imul":cs.n_imul_constraints(),
            "zero":cs.n_zero_constraints(), "bmul":cs.n_bmul_constraints()},
        "basefold":basefold, "binius_ligerito":ligerito,
    })
}

fn ligerito_report(prepared: &bitz::binius_ligerito::Prepared) -> Value {
    let security = prepared.security();
    let oracles: Vec<_> = prepared
        .oracle_specs()
        .iter()
        .enumerate()
        .map(|(index, spec)| {
            let pcs = prepared.opener(index);
            json!({
                "index": index, "log_packed_words": spec.log_msg_len,
                "configuration": pcs.config(),
                "round0_grinding_bits": pcs.ood_grinding_bits(),
                "round0_bits_before_grinding": pcs.ood_round_bits(),
            })
        })
        .collect();
    json!({
                "supported":true,
                "log_inv_rate": bitz::binary_pcs::LOG_INV_RATE,
                "target_bits": security.target_bits,
                "algebraic_bits": security.algebraic_bits,
                "component_bits": prepared.component_bits(),
                "terms": security.terms.iter().map(|term| json!({
                    "name":term.name, "error_bound":term.error_bound,
                })).collect::<Vec<_>>(),
                "oracles": oracles,
    })
}
