#!/usr/bin/env python3
"""Alternate prebuilt SHA/P256 prover snapshots; retain verified samples.

Run under scripts/bench_gate.py. Allocation-instrumented binaries must not be
used here. Each block is a fresh process with the worker's own warmup.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import statistics
from bench_support import run_checked, clean_environment, file_hash, write_json


from bench_statistics import paired_interval, classify_interval, timing_ratios

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exponents', type=int, nargs='+', default=[7, 10])
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 10])
    parser.add_argument('--methods', nargs='+', default=['bitz-split'])
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--targets', type=int, nargs='+', choices=[100, 128], default=[100])
    parser.add_argument('--baseline-env', action='append', default=[])
    parser.add_argument('--candidate-env', action='append', default=[])
    parser.add_argument('--schedule', choices=['auto','l2','l4','l8'], default='auto')
    parser.add_argument('--require-transcripts', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binaries = {k: getattr(args, k).resolve(strict=True) for k in ['baseline', 'candidate']}
    clean_env = clean_environment(os.environ)
    manifest = dict(binaries={k: dict(path=str(v), sha256=file_hash(v))
                              for k, v in binaries.items()},
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    affinity=sorted(os.sched_getaffinity(0)))
    write_json(args.output/'manifest.json',manifest)
    summaries = []
    for method, exponent, threads, target, seed in itertools.product(
            args.methods, args.exponents, args.threads, args.targets, args.seeds or [args.seed]):
        samples = {k: [] for k in binaries}
        cold = {k: [] for k in binaries}
        blocks = {k: [] for k in binaries}
        digest = None
        transcripts = None
        for block in range(args.blocks):
            for variant in (list(binaries) if block % 2 == 0 else list(binaries)[::-1]):
                name = f'{method}-i{exponent}-t{threads}-target{target}-seed{seed}-b{block}-{variant}'
                env = dict(clean_env, BITZ_LIG_PROFILE='custom:1:4', F2_FOREST_SCHEDULE=args.schedule,
                           RAYON_NUM_THREADS=str(threads), HARDWARE_CONCURRENCY=str(threads))
                # The target-100 tuning profile is not valid at 128 bits.
                # Use the benchmark's validated default for target 128.
                if target == 128:
                    env.pop('BITZ_LIG_PROFILE')
                env.update(dict(item.split('=', 1) for item in getattr(args, variant + '_env')))
                available = sorted(os.sched_getaffinity(0))
                cpus = ','.join(map(str, available[:threads]))
                command = ['taskset', '-c', cpus, str(binaries[variant]), '--method', method,
                           '--r', str(exponent), '--c', '0', '--target', str(target),
                           '--threads', str(threads), '--reps', str(args.reps), '--seed', str(seed)]
                with (args.output / (name+'.stdout')).open('w') as out, (args.output / (name+'.stderr')).open('w') as err:
                    run_checked(command, env=env, stdout=out, stderr=err, timeout=600)
                rows = [json.loads(line) for line in (args.output / (name+'.stdout')).read_text().splitlines()
                        if line.startswith('{')]
                rows = [{**r, **r.get('measurements', {})} for r in rows]
                assert len(rows) == args.reps + 1 and all(r['verified'] for r in rows), name
                for row in rows:
                    digest = digest or row['proof_digest']
                    assert row['proof_digest'] == digest, f'proof bytes changed: {name}'
                    state = [row.get(key) for key in ['prover_transcript', 'verifier_transcript']]
                    if args.require_transcripts:
                        assert all(state), f'missing transcript fingerprints: {name}'
                    transcripts = transcripts or state
                    assert state == transcripts, f'transcript changed: {name}'
                for row in rows:
                    phases = dict(row['phases_seconds'])
                    row['gkr_ms'] = 1000 * phases['mc:forest']
                    row['grid_ms'] = 1000 * phases.get('eqf:grid', 0)
                metrics = ['e2e_prover_ms', 'prove_ms', 'gkr_ms', 'grid_ms', 'verify_ms']
                first = [r for r in rows if r['trial'] == 'warmup']
                assert len(first) == 1, name
                cold[variant].extend(first)
                rows = [r for r in rows if r['trial'] == 'sample']
                medians = {m: statistics.median(r[m] for r in rows) for m in metrics}
                medians.update({'cold_' + m: first[0][m] for m in metrics})
                samples[variant].extend(rows)
                blocks[variant].append(medians)
                print(name, {m: round(v, 3) for m, v in medians.items()}, flush=True)
        summary = dict(method=method, exponent=exponent, threads=threads, target=target, seed=seed, proof_digest=digest, transcripts=transcripts, metrics={})
        for metric in metrics + ['cold_' + m for m in metrics]:
            source = cold if metric.startswith('cold_') else samples
            field = metric.removeprefix('cold_')
            ratios = timing_ratios([b[metric] for b in blocks['baseline']],
                                   [b[metric] for b in blocks['candidate']],
                                   diagnostic=field == 'grid_ms')
            interval = paired_interval(ratios) if ratios else None
            summary['metrics'][metric] = dict(
                baseline_ms=statistics.median(r[field] for r in source['baseline']),
                candidate_ms=statistics.median(r[field] for r in source['candidate']),
                paired_ratios=ratios, ratio_ci95=interval)
            if ratios:
                summary['metrics'][metric]['nonregression'] = classify_interval(interval)
        summaries.append(summary)
        write_json(args.output/'summary.json',summaries)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
