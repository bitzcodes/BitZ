"""Guard against accepting uncertain or invalid performance measurements."""
import unittest

from bench_statistics import classify_interval, paired_interval, timing_ratios
from qualify_gkr import evaluate


class PerformanceGateTests(unittest.TestCase):
    def test_missing_coverage_cannot_pass(self):
        result = evaluate([], [])
        self.assertFalse(result['passed'])
        self.assertTrue(result['missing'])

    def test_tuning_blocks_cannot_qualify(self):
        row = dict(method='bitz-split', exponent=10, threads=10, target=100, seed=31,
                   metrics={name: dict(paired_ratios=[0.9] * 6) for name in
                            ['gkr_ms', 'e2e_prover_ms', 'prove_ms', 'cold_e2e_prover_ms', 'cold_prove_ms', 'cold_gkr_ms']})
        with self.assertRaisesRegex(ValueError, '24 blocks'):
            evaluate([row], [])

    def test_removed_optional_phase_does_not_hide_missing_proving_time(self):
        self.assertEqual(timing_ratios([1, 2], [0, 0], diagnostic=True), [])
        with self.assertRaises(ValueError):
            timing_ratios([1, 2], [0, 0])
        with self.assertRaises(ValueError):
            timing_ratios([1, 2], [1])

    def test_uncertainty_is_not_a_pass(self):
        self.assertEqual(classify_interval([0.99, 1.01]), 'inconclusive')
        self.assertEqual(classify_interval([1.001, 1.01]), 'regression')
        self.assertEqual(classify_interval([0.98, 1.0]), 'pass')

    def test_two_percent_is_a_separate_gate(self):
        self.assertEqual(classify_interval([0.975, 0.985], 0.98), 'inconclusive')
        self.assertEqual(classify_interval([0.97, 0.98], 0.98), 'pass')

    def test_invalid_or_unpaired_data_is_rejected(self):
        for ratios in [[], [1], [1, 0], [1, -1], [1, float('nan')], [1, float('inf')]]:
            with self.assertRaises(ValueError):
                paired_interval(ratios)

    def test_exact_repeated_ratio(self):
        low, high = paired_interval([0.97] * 6)
        self.assertAlmostEqual(low, 0.97)
        self.assertAlmostEqual(high, 0.97)


if __name__ == '__main__':
    unittest.main()
