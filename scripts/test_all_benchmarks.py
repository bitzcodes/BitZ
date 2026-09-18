"""Exercise the complete shell workflow with lightweight command stand-ins."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AllBenchmarks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "workspace with spaces"
        (self.root / "scripts").mkdir(parents=True)
        self.bin = self.root / "fake-bin"
        self.bin.mkdir()
        self.trace = self.root / "commands.jsonl"
        self.output = self.root / "results with spaces"
        shutil.copy2(ROOT / "scripts/run_all_benchmarks.sh", self.root / "scripts/run_all_benchmarks.sh")
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        TEST_ROOT=str(self.root), TEST_TRACE=str(self.trace),
                        CARGO_TARGET_DIR="/reused/build-cache")
        driver = r'''
import json, os, sys, subprocess
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
root = Path(os.environ['TEST_ROOT'])
with open(os.environ['TEST_TRACE'], 'a') as stream:
    stream.write(json.dumps(dict(name=name,args=args,cargo_target_dir=os.environ.get('CARGO_TARGET_DIR'),
        layout=os.environ.get('BITZ_SHA_PRODUCT_TS'), reps=os.environ.get('BITZ_BENCH_REPS'),
        rate=os.environ.get('BITZ_BINIUS_LOG_INV_RATE'),threads=os.environ.get('RAYON_NUM_THREADS'),
        inherited_shape=os.environ.get('BITZ_SHA_LOG2S'), inherited_reps=os.environ.get('BITZ_SHA_REPS'),
        lock=os.environ.get('BITZ_BENCH_LOCK'))) + '\n')
if name == 'python3':
    script = args[0]
    if script in ['-c','-']:
        raise SystemExit(subprocess.call([sys.executable,*args]))
    if script.endswith('bench_gate.py'):
        raise SystemExit(subprocess.call(args[args.index('--')+1:]))
    if script.endswith(os.environ.get('FAIL_STEP','<none>')):
        raise SystemExit(7)
    if script.endswith('rss_sampler.py'):
        Path(args[args.index('--output')+1]).write_text('sample\n')
        raise SystemExit(subprocess.call(args[args.index('--')+1:]))
elif name == 'cargo' and '--message-format=json' in args:
    print(json.dumps(dict(reason='compiler-artifact', target=dict(name='hybrid_u32_sha256',
        kind=['bench'],src_path=str(root/'benches/hybrid_u32_sha256.rs')),
        executable=str(root/'fake-bin/hybrid'))))
elif name == 'hybrid':
    Path(args[args.index('--results-dir')+1]).mkdir(parents=True)
'''
        for name in ["python3", "cargo", "rustup", "hybrid"]:
            path = self.bin / name
            path.write_text(f"#!{sys.executable}\n" + driver)
            path.chmod(0o755)
        (self.root / "scripts/install_trace_processor.sh").write_text("#!/usr/bin/env bash\nexit 0\n")

    def run_wrapper(self, *args, **environment):
        return subprocess.run(["bash", str(self.root / "scripts/run_all_benchmarks.sh"),
                               "--output", str(self.output), *args],
                              cwd=self.root, env=self.env | environment,
                              capture_output=True, text=True, timeout=20)

    def commands(self):
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def test_full_run_preserves_seven_steps_and_takes_one_outer_lock(self):
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        scripts = [c for c in commands if c['name'] == 'python3' and c['args'][0].startswith('scripts/')]
        names = [c['args'][0] for c in scripts]
        self.assertEqual(names.count('scripts/bench_gate.py'), 1)
        self.assertEqual(names, [
            'scripts/bench_gate.py', 'scripts/materialize_vendors.py',
            'scripts/run_sha256_ecdsa_compare.py', 'scripts/run_sha256_chain_compare.py',
            'scripts/run_multiplication_benchmarks.py', 'scripts/run_matched_multiswap_campaign.py',
            'scripts/hybrid_table.py'])
        multiplication = next(c['args'] for c in scripts if c['args'][0].endswith('run_multiplication_benchmarks.py'))
        self.assertIn('--no-gate', multiplication)
        self.assertNotIn('--exponents', multiplication)
        hybrid = [c for c in commands if c['name'] == 'hybrid']
        self.assertEqual(len(hybrid), 12)
        for c in hybrid:
            self.assertEqual(c['args'][c['args'].index('--shapes')+1], '15:7,16:8,17:9,18:10,19:11,20:12')
            self.assertEqual(c['args'][c['args'].index('--iterations')+1], '5')
        self.assertEqual({c['rate'] for c in hybrid}, {'1','3'})
        self.assertEqual({c['threads'] for c in hybrid}, {'1','10'})
        self.assertTrue(all(c['cargo_target_dir'] == '/reused/build-cache' for c in commands))
        self.assertEqual([f'[{n}/7]' in result.stdout for n in range(1,8)], [True]*7)
        self.assertIn('All seven campaigns completed.', result.stdout)

    def test_smoke_reduces_sizes_and_samples_but_keeps_backends_rates_and_threads(self):
        result = self.run_wrapper('--smoke', '--no-gate')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        self.assertFalse(any('scripts/bench_gate.py' in c['args'] for c in commands))
        for script in ['run_sha256_ecdsa_compare.py', 'run_sha256_chain_compare.py', 'run_multiplication_benchmarks.py']:
            args = next(c['args'] for c in commands if c['args'][0] == 'scripts/'+script)
            self.assertEqual(args[args.index('--reps')+1], '1')
            self.assertEqual(args[args.index('--threads')+1:args.index('--threads')+3], ['1','10'])
        hybrid = [c for c in commands if c['name'] == 'hybrid']
        self.assertEqual(len(hybrid), 24)
        self.assertEqual({c['args'][c['args'].index('--shapes')+1] for c in hybrid}, {'15:7','9:9'})
        self.assertTrue(all(c['args'][c['args'].index('--iterations')+1] == '1' for c in hybrid))
        layout = next(c for c in commands if 'sha256_product_layout' in c['args'])
        self.assertEqual((layout['layout'],layout['reps']), ('13','1'))
        self.assertIn('All seven campaigns completed.', result.stdout)

    def test_dry_run_never_starts_tools_or_creates_output(self):
        result = self.run_wrapper('--smoke', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.trace.exists())
        self.assertFalse(self.output.exists())
        self.assertIn('Outer lock:', result.stdout)
        self.assertIn('Dry-run complete; no campaigns were executed.', result.stdout)

    def test_existing_output_is_protected_before_starting_tools(self):
        self.output.mkdir()
        marker = self.output / 'keep'
        marker.write_text('unchanged')
        result = self.run_wrapper('--smoke')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.trace.exists())
        self.assertEqual(marker.read_text(), 'unchanged')

    def test_verification_failure_prevents_campaigns(self):
        result = self.run_wrapper('--smoke', FAIL_STEP='materialize_vendors.py')
        self.assertEqual(result.returncode, 7)
        self.assertFalse(self.output.exists())
        self.assertFalse(any(c['name'] in ['cargo','rustup','hybrid'] for c in self.commands()))
        self.assertNotIn('[1/7]', result.stdout)

    def test_failed_campaign_keeps_logs_and_cannot_claim_success(self):
        result = self.run_wrapper('--smoke', FAIL_STEP='run_sha256_chain_compare.py')
        self.assertEqual(result.returncode, 7)
        self.assertTrue((self.output/'sha256-chain.log').is_file())
        self.assertNotIn('[3/7]', result.stdout)
        self.assertNotIn('All seven campaigns completed.', result.stdout)

    def test_darwin_sampler_is_used_and_its_artifacts_are_moved(self):
        uname = self.bin / 'uname'
        uname.write_text('#!/usr/bin/env bash\nprintf "Darwin\\n"\n')
        uname.chmod(0o755)
        result = self.run_wrapper('--smoke', '--no-gate')
        self.assertEqual(result.returncode, 0, result.stderr)
        samplers = [c for c in self.commands() if c['args'][0] == 'scripts/rss_sampler.py']
        self.assertEqual(len(samplers), 24)
        self.assertEqual(len(list(self.output.glob('hybrid-*/*/peak-rss-and-swap.tsv'))), 24)

    def test_bad_arguments_are_rejected(self):
        for flags in [('--unknown',), ('--swap-grow-gb','nan'), ('--swap-grow-gb','0'), ('--output',)]:
            with self.subTest(flags=flags):
                self.assertEqual(self.run_wrapper(*flags).returncode, 2)
        self.assertFalse(self.trace.exists())

    def test_inherited_experiment_controls_cannot_change_the_campaigns(self):
        result = self.run_wrapper('--smoke', '--no-gate', BITZ_SHA_LOG2S='9', BITZ_SHA_REPS='99',
                                  BITZ_BENCH_REPS='99', RAYON_NUM_THREADS='44', BITZ_BENCH_LOCK='/custom/lock')
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in self.commands():
            self.assertIsNone(command['inherited_shape'])
            self.assertIsNone(command['inherited_reps'])
            self.assertEqual(command['lock'], '/custom/lock')
        layout = next(c for c in self.commands() if 'sha256_product_layout' in c['args'])
        self.assertEqual((layout['layout'],layout['reps'],layout['threads']), ('13','1','10'))


if __name__ == '__main__':
    unittest.main()
