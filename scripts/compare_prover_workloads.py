#!/usr/bin/env python3
"""Paired regression runs for the existing multiplication, SHA and MultiSwap benches.

Each manifest maps bench names to prebuilt executables. Run under bench_gate.py;
use matching features/compiler flags and no allocation instrumentation.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
from bench_support import run_checked, clean_environment, file_hash, write_json, address_space_limit
from mul_results import load
import subprocess
from bench_statistics import paired_interval, classify_interval, sample_statistics


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]


def read_samples(kind, directory, reps, trial_kind="sample"):
    count = reps if trial_kind == "sample" else 1
    lines = [line.strip() for line in (directory / 'stdout').read_text().splitlines()]
    if kind in ('mul', 'u32'):
        cases = load(directory / 'results')
        if len(cases) != 1 or cases[0].status != 'measured':
            raise ValueError('expected one verified multiplication case')
        selected = [r for r in cases[0].records if r['kind'] == trial_kind]
        if len(selected) != count:
            raise ValueError('missing multiplication samples')
        result = []
        for record in selected:
            m = record['metrics']
            row = dict(prove_ms=m['online_prover_ms'], verify_ms=m['verify_ms'],
                       proof_bytes=m['proof_bytes'], gkr_ms=m['prove/mc:forest_ms'])
            if 'witness_to_proof_ms' in m:
                row['e2e_ms'] = m['witness_to_proof_ms']
            result.append(row)
        return result
    if kind == 'multiswap':
        runs = [r for r in jsonl(directory/'multiswap.jsonl') if r['record'] == 'run' and r['trial']['kind'] == trial_kind]
        assert len(runs) == count and all(r['validation']['proof_verified'] for r in runs)
        return [dict(e2e_ms=int(r['measurements_ns']['application_total'])/1e6,
                     prove_ms=int(r['measurements_ns']['online_prover'])/1e6,
                     verify_ms=int(r['measurements_ns']['verification'])/1e6,
                     gkr_ms=int(r['measurements_ns']['merged_forest_gkr'])/1e6,
                     proof_bytes=r['artifacts']['proof_bytes']) for r in runs]
    trials = [json.loads(line.split(' ', 1)[1]) for line in lines if line.startswith('PROVER_TRIAL ')]
    selected = [r for r in trials if r['trial'] == trial_kind]
    if len(selected) != count or not all(r['verified'] for r in selected):
        raise ValueError('missing verified prover trials')
    return selected


def mul_command(case, reps, output=None):
    """Serialize one Rust-resolved case; configuration expansion belongs to Rust."""
    f = case['bitz']
    command = ['proof', '--workload', case['workload'], '--backends', 'bitz',
               '--log-n', str(case['log_n']), '--threads', str(case['threads']),
               '--seed', str(case['seed']), '--w', str(f['w']), '--split='+str(f['split']),
               '--bitz-profile', str(f['profile']), '--ligerito', f['ligerito'],
               '--gkr-schedule', f['gkr_schedule'], '--reps', str(reps), '--warmups', '1',
               '--proof-fingerprints']
    if output is not None:
        command += ['--out', str(output)]
    return command


def multiplication_cases(args, binaries, env):
    cases = []
    for kind, target, workloads, widths in [('mul','mul_compare',','.join(args.workloads),'1'),
                                            ('u32','mul_bitz','u32-full','1,8')]:
        if kind not in args.kinds:
            continue
        command = ['proof','--workload',workloads,'--backends','bitz','--w',widths,
                   '--log-n',','.join(map(str,args.exponents)), '--threads',','.join(map(str,args.threads)),
                   '--seed',str(args.seed),'--ligerito','custom:1:4','--gkr-schedule',args.schedule,'--dry-run']
        previews = {}
        for variant in binaries:
            previews[variant] = json.loads(subprocess.check_output([binaries[variant][target]['path'],*command],env=env,text=True))
        left = [j['case'] for j in previews['baseline']]
        right = [j['case'] for j in previews['candidate']]
        if left != right or any(j['skip'] for jobs in previews.values() for j in jobs):
            raise ValueError('multiplication revisions resolve different or unsupported cases')
        for case in left:
            name = (f"full-u32-w{case['bitz']['w']}" if kind == 'u32' else case['workload']) + f"-n{case['log_n']}"
            cases.append((kind,target,name,{},case))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kinds', nargs='+', default=['mul', 'sha', 'u32', 'multiswap'])
    parser.add_argument('--workloads', nargs='+', choices=['u32-mod32', 'u64', 'u128'], default=['u32-mod32', 'u64', 'u128'])
    parser.add_argument('--exponents', type=int, nargs='+', default=[15, 19])
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 10])
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--sha-exponents', type=int, nargs='+', default=[7, 10, 12])
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--baseline-env', action='append', default=[])
    parser.add_argument('--candidate-env', action='append', default=[])
    parser.add_argument('--require-cold-and-fingerprints', action='store_true')
    parser.add_argument('--schedule', choices=['auto', 'l2', 'l4', 'l8'], default='auto')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifests = {v: json.loads(getattr(args, v).read_text()) for v in ['baseline', 'candidate']}
    binaries = {v: {name: dict(path=str(Path(path).resolve()), sha256=file_hash(path))
                    for name, path in manifest.items()} for v, manifest in manifests.items()}
    write_json(args.output/'manifest.json', dict(binaries=binaries, args={k:str(v) for k,v in vars(args).items()}, affinity=sorted(os.sched_getaffinity(0))))
    clean = clean_environment(os.environ)
    cases = multiplication_cases(args,binaries,clean)
    if 'sha' in args.kinds:
        cases += [('sha', 'sha256_chain', f'sha-n{n}', dict(BITZ_BENCH_SHAPES=str(n)),None) for n in args.sha_exponents]
    if 'multiswap' in args.kinds:
        cases += [('multiswap', 'multiswap', f'multiswap-b{b}', dict(BITZ_BENCH_SHAPES='0', BITZ_MULTISWAP_BATCH_COUNT=str(b), BITZ_BENCH_LAMBDA='114'),None) for b in args.batches]
    results = []
    for kind, bench, case, case_env, resolved in cases:
        for threads in ([resolved['threads']] if resolved else args.threads):
            blocks = {v: [] for v in manifests}
            sample_sizes = None
            first_size = None
            expected_fingerprints = None
            schedules = {}
            for block in range(args.blocks):
                for variant in (list(manifests) if block % 2 == 0 else list(manifests)[::-1]):
                    directory = (args.output/f'{case}-t{threads}-b{block}-{variant}').resolve()
                    directory.mkdir()
                    env = dict(clean, BITZ_LIG_PROFILE='custom:1:4', F2_FOREST_SCHEDULE=args.schedule,
                               RAYON_NUM_THREADS=str(threads), HARDWARE_CONCURRENCY=str(threads), BITZ_BENCH_SEED=str(args.seed),
                               BITZ_BENCH_REPS=str(args.reps), BITZ_BENCH_LAMBDA='100', BITZ_BENCH_PASS='latency',
                               BITZ_BENCH_PHASE_SAMPLES='1', BITZ_BENCH_PROOF_FINGERPRINT='1',
                               BITZ_MULTISWAP_TRACE_PATH=str(directory/'multiswap.jsonl'))
                    env.update(case_env)
                    env.update(item.split('=', 1) for item in getattr(args, variant + '_env'))
                    cpus = ','.join(map(str, sorted(os.sched_getaffinity(0))[:threads]))
                    command = ['taskset', '-c', cpus, binaries[variant][bench]['path']]
                    if resolved:
                        command += mul_command(resolved,args.reps,directory/'results')
                    with (directory/'stdout').open('w') as out, (directory/'stderr').open('w') as err:
                        run_checked(command, env=env, stdout=out, stderr=err, timeout=1800,
                                    preexec_fn=address_space_limit(48))
                    fingerprints = [json.loads(line.split(' ', 1)[1]) for line in
                                    (directory/'stdout').read_text().splitlines()
                                    if line.startswith('PROOF_FINGERPRINT ')]
                    if resolved:
                        campaign = load(directory/'results')[0]
                        if campaign.job['case'] != resolved:
                            raise ValueError('worker changed the resolved case')
                        fingerprints = [r['fingerprint'] for r in campaign.records if r['kind'] in ('warmup','sample')]
                    if args.require_cold_and_fingerprints:
                        assert fingerprints, f'missing proof fingerprint: {directory}'
                    if fingerprints:
                        assert len(fingerprints) == args.reps + 1
                        # SHA chains deliberately use a different seed for
                        # each trial. Compare corresponding trials across
                        # variants and blocks, not different inputs in a run.
                        if expected_fingerprints is None:
                            expected_fingerprints = fingerprints
                        assert fingerprints == expected_fingerprints, f'proof/transcript changed: {case}'
                    if resolved:
                        actual = campaign.effective['gkr_schedules']
                    else:
                        actual = [json.loads(line.split(' ', 1)[1]) for line in
                                  (directory/'stdout').read_text().splitlines()
                                  if line.startswith('GKR_SCHEDULES ')]
                    if variant in schedules and schedules[variant] != actual:
                        raise ValueError('resolved schedules changed across blocks')
                    schedules[variant] = actual
                    samples = read_samples(kind, directory, args.reps)
                    sizes = [sample['proof_bytes'] for sample in samples]
                    if sample_sizes is None:
                        sample_sizes = sizes
                    assert sizes == sample_sizes, f'proof size changed: {case}'
                    for sample in samples:
                        assert sample['gkr_ms'] > 0, 'missing GKR measurement'
                    write_json(directory/'samples.json',samples)
                    medians = {m:sample_statistics([r[m] for r in samples])['median'] for m in ['prove_ms','gkr_ms','verify_ms', *(['e2e_ms'] if 'e2e_ms' in samples[0] else [])]}
                    first = read_samples(kind, directory, args.reps, 'warmup')
                    if args.require_cold_and_fingerprints:
                        assert first, f'missing first-proof metrics: {directory}'
                    if first:
                        if first_size is None:
                            first_size = first[0]['proof_bytes']
                        assert first[0]['proof_bytes'] == first_size, f'first proof size changed: {case}'
                        medians.update({'cold_' + m: first[0][m] for m in ['prove_ms','gkr_ms','verify_ms', *(['e2e_ms'] if 'e2e_ms' in first[0] else [])]})
                    blocks[variant].append(medians)
                    print(directory.name, {m:round(v,3) for m,v in medians.items()}, flush=True)
            result = dict(case=case, configuration=resolved, schedules=schedules, total_metric='prove_ms' if kind == 'u32' else 'e2e_ms', threads=threads, seed=args.seed, sample_proof_bytes=sample_sizes,
                          first_proof_bytes=first_size, fingerprints=expected_fingerprints, metrics={})
            common_metrics = set.intersection(*(set(row) for rows in blocks.values() for row in rows))
            for metric in sorted(common_metrics):
                ratios = [b[metric]/a[metric] for a,b in zip(blocks['baseline'],blocks['candidate'])]
                interval = paired_interval(ratios)
                result['metrics'][metric] = dict(baseline_ms=statistics.median(b[metric] for b in blocks['baseline']),
                    candidate_ms=statistics.median(b[metric] for b in blocks['candidate']), paired_ratios=ratios, ratio_ci95=interval,
                    nonregression=classify_interval(interval))
            results.append(result)
            write_json(args.output/'summary.json',results)


if __name__ == '__main__':
    main()
