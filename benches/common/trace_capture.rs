//! Reporting projections over completed Perfetto intervals; no clocks or collector.
pub(crate) use bitz::observability::Interval as CapturedSpan;

fn required_span<'a>(raw: &'a [CapturedSpan], component: &str) -> &'a CapturedSpan {
    let mut spans = raw
        .iter()
        .filter(|s| s.component.as_deref() == Some(component));
    let span = spans
        .next()
        .unwrap_or_else(|| panic!("missing {component} span"));
    assert!(
        spans.next().is_none(),
        "multiple {component} spans in one trial"
    );
    assert!(span.end_ns >= span.start_ns, "reversed {component} span");
    span
}

/// Completed trial scopes. Keep each operation's own endpoints rather than
/// charging the enclosing scopes' entry/exit overhead to witness or verifier.
pub(crate) struct TrialScopes<'a> {
    pub(crate) verified: &'a CapturedSpan,
    pub(crate) witness_to_proof: &'a CapturedSpan,
    pub(crate) witness: &'a CapturedSpan,
    pub(crate) verification: &'a CapturedSpan,
}

impl<'a> TrialScopes<'a> {
    pub(crate) fn from_spans(raw: &'a [CapturedSpan], prefix: &str) -> Self {
        let verified = required_span(raw, &format!("{prefix}.verified-trial"));
        let witness_to_proof = required_span(raw, &format!("{prefix}.witness-to-proof"));
        let witness = required_span(raw, &format!("{prefix}.witness-evaluation"));
        let verification = required_span(raw, &format!("{prefix}.verification"));
        assert!(
            verified.start_ns <= witness_to_proof.start_ns
                && witness_to_proof.start_ns <= witness.start_ns
                && witness.end_ns <= witness_to_proof.end_ns
                && witness_to_proof.end_ns <= verification.start_ns
                && verification.end_ns <= verified.end_ns,
            "trial spans are not nested/sequenced correctly"
        );
        Self {
            verified,
            witness_to_proof,
            witness,
            verification,
        }
    }
}

/// Reporting attribution, not another timer: Round 0 is opening work even
/// though it executes inside the PIOP prefix. Later oracle commits remain
/// attributed to PIOP, matching the existing benchmark metric definitions.
pub(crate) struct BiniusLigeritoPhases {
    pub(crate) commit: (u64, u64),
    pub(crate) piop: Vec<(u64, u64)>,
    pub(crate) opening: Vec<(u64, u64)>,
}

impl BiniusLigeritoPhases {
    pub(crate) fn from_spans(raw: &[CapturedSpan]) -> Self {
        let matching = |component| {
            raw.iter()
                .filter(move |s| s.component.as_deref() == Some(component))
        };
        let required = |component| {
            let span = required_span(raw, component);
            (span.start_ns, span.end_ns)
        };
        let prefix = required("binius-ligerito.piop");
        let commit = required("binius-ligerito.witness-commit");
        let final_opening = required("binius-ligerito.opening");
        assert!(
            prefix.1 <= final_opening.0,
            "opening precedes the PIOP prefix end"
        );
        let mut opening: Vec<_> = matching("binius-ligerito.round0")
            .map(|s| (s.start_ns, s.end_ns))
            .collect();
        assert!(!opening.is_empty(), "missing binius-ligerito.round0 span");

        // Subtract the union, preserving gaps and actual placement. In
        // particular, never slide Round 0 past the constraint reductions.
        let mut excluded = opening.clone();
        excluded.push(commit);
        excluded.sort_unstable();
        let mut cursor = prefix.0;
        let mut piop = Vec::new();
        for (start, end) in excluded {
            assert!(
                prefix.0 <= start && start <= end && end <= prefix.1,
                "commit/Round 0 outside the PIOP prefix"
            );
            if cursor < start {
                piop.push((cursor, start));
            }
            cursor = cursor.max(end);
        }
        if cursor < prefix.1 {
            piop.push((cursor, prefix.1));
        }
        opening.push(final_opening);
        opening.sort_unstable();
        Self {
            commit,
            piop,
            opening,
        }
    }
}

#[cfg(test)]
pub(crate) mod phase_tests {
    use super::*;

    fn fixture() -> Vec<CapturedSpan> {
        [
            ("binius-ligerito.piop", 10, 100),
            ("binius-ligerito.witness-commit", 20, 30),
            ("binius-ligerito.round0", 30, 40),
            ("binius-ligerito.oracle-commit", 55, 65),
            ("binius-ligerito.round0", 65, 75),
            ("binius-ligerito.round0", 70, 80),
            ("binius-ligerito.opening", 105, 130),
        ]
        .into_iter()
        .enumerate()
        .map(|(id, (component, start_ns, end_ns))| CapturedSpan {
            track_id: 0,
            depth: 0,
            id: id as u64,
            parent: None,
            name: component.to_owned(),
            component: Some(component.to_owned()),
            start_ns,
            end_ns,
        })
        .collect()
    }

    pub(crate) fn trial_fixture() -> Vec<CapturedSpan> {
        let mut raw = fixture();
        raw.extend(
            [
                ("binius-ligerito.verified-trial", 0, 160),
                ("binius-ligerito.witness-to-proof", 1, 140),
                ("binius-ligerito.witness-evaluation", 2, 10),
                ("binius-ligerito.verification", 145, 155),
            ]
            .into_iter()
            .enumerate()
            .map(|(id, (component, start_ns, end_ns))| CapturedSpan {
                track_id: 0,
                depth: 0,
                id: 100 + id as u64,
                parent: None,
                name: component.to_owned(),
                component: Some(component.to_owned()),
                start_ns,
                end_ns,
            }),
        );
        raw
    }

    #[test]
    fn trial_scopes_keep_distinct_endpoints() {
        let raw = trial_fixture();
        let trial = TrialScopes::from_spans(&raw, "binius-ligerito");
        assert_eq!((trial.verified.start_ns, trial.verified.end_ns), (0, 160));
        assert_eq!(
            (
                trial.witness_to_proof.start_ns,
                trial.witness_to_proof.end_ns
            ),
            (1, 140)
        );
        assert_eq!((trial.witness.start_ns, trial.witness.end_ns), (2, 10));
        assert_eq!(
            (trial.verification.start_ns, trial.verification.end_ns),
            (145, 155)
        );
    }

    #[test]
    #[should_panic(expected = "missing binius-ligerito.verification span")]
    fn incomplete_trial_is_rejected() {
        let mut raw = trial_fixture();
        raw.pop();
        TrialScopes::from_spans(&raw, "binius-ligerito");
    }

    #[test]
    #[should_panic(expected = "trial spans are not nested/sequenced correctly")]
    fn verification_cannot_overlap_proof_production() {
        let mut raw = trial_fixture();
        raw.last_mut().unwrap().start_ns = 139;
        TrialScopes::from_spans(&raw, "binius-ligerito");
    }

    #[test]
    fn round0_keeps_its_position_and_is_excluded_once_from_piop() {
        let mut raw = fixture();
        raw.reverse(); // Collection order must not determine the timeline.
        let phases = BiniusLigeritoPhases::from_spans(&raw);
        assert_eq!(phases.commit, (20, 30));
        assert_eq!(phases.piop, [(10, 20), (40, 65), (80, 100)]);
        assert_eq!(phases.opening, [(30, 40), (65, 75), (70, 80), (105, 130)]);
    }

    #[test]
    #[should_panic(expected = "missing binius-ligerito.opening span")]
    fn missing_phase_is_not_zero() {
        let mut raw = fixture();
        raw.pop();
        BiniusLigeritoPhases::from_spans(&raw);
    }

    #[test]
    #[should_panic(expected = "multiple binius-ligerito.piop spans")]
    fn mixed_trials_are_rejected() {
        let mut raw = fixture();
        raw.push(raw[0].clone());
        BiniusLigeritoPhases::from_spans(&raw);
    }

    #[test]
    #[should_panic(expected = "commit/Round 0 outside the PIOP prefix")]
    fn invalid_interval_is_not_clipped() {
        let mut raw = fixture();
        raw[2].end_ns = 200;
        BiniusLigeritoPhases::from_spans(&raw);
    }
}
