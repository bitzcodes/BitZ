#!/usr/bin/env python3
"""Apple Silicon inner-sumcheck A/B campaign. Run under scripts/bench_gate.py.

Frozen executables are supplied by --artifacts/{baseline,candidate}. Each cell
uses independent processes, alternating AB/BA order. A two-sided 95% Student-t
interval on paired log median ratios classifies the 1% gate; overlapping cells
remain inconclusive. Failed cells are recorded, never silently dropped.
"""
import argparse, json, math, os, platform, re, statistics, time
from bench_support import file_hash, filtered_environment, run_process, write_json
from pathlib import Path

def cases(extended=False):
    result = []
    for threads in [1, 10]:
        for target in [100, 128]:
            for mode in ['split', 'all']:
                result.append(dict(id=f'ecdsa-{mode}-{target}-t{threads}', exe='sha256_ecdsa', args=['3', mode, str(target)], threads=threads, env={}))
            result.append(dict(id=f'sha-chain-{target}-t{threads}', exe='sha256_chain', args=[], threads=threads, env={'BITZ_BENCH_SHAPES': '7', 'BITZ_BENCH_LAMBDA': str(target)}))
        for width in [1, 8]:
            result.append(dict(id=f'u32-w{width}-t{threads}', exe='mul_bitz', args=['proof', '--workload', 'u32-full', '--log-n', '15', '--w', str(width)], threads=threads, env={}))
        for width in [64, 128]:
            result.append(dict(id=f'u{width}-t{threads}', exe='mul_bitz', args=['proof', '--workload', f'u{width}', '--log-n', '15'], threads=threads, env={}))
        for batch in [1, 2]:
            result.append(dict(id=f'multiswap-b{batch}-t{threads}', exe='multiswap', args=[], threads=threads, env={'BITZ_BENCH_SHAPES': '0', 'BITZ_MULTISWAP_BATCH_COUNT': str(batch), 'BITZ_BENCH_LAMBDA': '114'}))
    if extended:
        for old in list(result):
            if old['exe'] in ['mul_bitz', 'sha256_chain']:
                c = {**old, 'env': dict(old['env']), 'args': list(old['args'])}
                if c['exe'] == 'sha256_chain':
                    c['env']['BITZ_BENCH_SHAPES'] = '9'
                else:
                    c['args'][c['args'].index('--log-n') + 1] = '17'
                c['id'] += '-large'
                result.append(c)
    return result

def json_lines(path):
    with open(path) as f:
        for line in f:
            if line.startswith('{'):
                yield json.loads(line)

def parse_run(case, run):
    out = (run / 'stdout.log').read_text()
    err = (run / 'stderr.log').read_text()
    samples = []
    proof = []
    if case['exe'] == 'sha256_ecdsa':
        for r in json_lines(run / 'stdout.log'):
            if r.get('trial') == 'sample':
                assert r['verified']
                samples.append({k: r[k] for k in ['prove_ms', 'verify_ms', 'witness_ms', 'commit_ms']})
                proof.append(r['proof_bytes'])
    elif case['exe'] == 'mul_bitz':
        from mul_results import load
        cases = load(run / 'mul')
        assert len(cases) == 1
        for r in cases[0].records:
            if r['kind'] == 'sample':
                m = r['metrics']
                samples.append({**m, 'prove_ms': m['online_prover_ms'], **{'phase:' + k: v for k, v in m.items() if k.startswith('prove/')}})
                proof.append(m['proof_bytes'])
    elif case['exe'] == 'multiswap':
        for r in json_lines(run / 'trace.jsonl'):
            if r.get('record') == 'run' and r['trial']['kind'] == 'sample':
                assert r['validation']['proof_verified']
                m = r['measurements_ns']
                samples.append({k: float(v) / 1000000.0 for k, v in m.items()})
                proof.append(r['artifacts']['proof_bytes'])
    else:
        for line in out.splitlines():
            fields = dict(re.findall('(\\w+)=([^ ]+)', line))
            if line.lstrip().startswith('SAMPLE '):
                assert fields['verified'] == 'true'
                samples.append({k: float(v) for k, v in fields.items() if k.endswith('_ms')})
            if line.lstrip().startswith('BENCH ') or line.lstrip().startswith('RESULT '):
                if 'proof_bytes' in fields:
                    proof.append(int(fields['proof_bytes']))
    if case['exe'] == 'multiswap':
        for s in samples:
            s['prove_ms'] = s['online_prover']
            s['verify_ms'] = s['verification']
            s['witness_ms'] = s['witness_generation']
    phase_rows = []
    if case['exe'] == 'sha256_ecdsa':
        phase_rows = [dict(r['prove_phases_seconds']) for r in json_lines(run / 'stdout.log') if r.get('trial') == 'sample']
    else:
        phase_rows = [dict(json.loads(line.split(' ', 1)[1])) for line in out.splitlines() if line.startswith('REGRESSION_PHASES ')]
    if phase_rows:
        assert len(phase_rows) == len(samples), 'phase/sample count mismatch'
        for sample, phases in zip(samples, phase_rows):
            sample.update({'phase:' + label: seconds * 1000 for label, seconds in phases.items()})
    if case['exe'] == 'mul_bitz':
        from mul_results import load
        raw = load(run / 'mul')[0].records
        heap_rows = [r for r in raw if r['kind'] == 'heap']
        if heap_rows:
            return dict(samples=[], proof_bytes=[], rss_bytes=None, heap_peak_bytes=heap_rows[0]['metrics']['peak_heap_bytes'])
    assert samples, 'no measured verified samples'
    assert proof and all((p > 0 for p in proof)), 'missing proof-size evidence'
    rss = re.search('(\\d+)\\s+maximum resident set size', err)
    heap = re.search('HEAP_RUN peak_bytes=(\\d+) live_bytes=(\\d+)', err)
    return dict(samples=samples, proof_bytes=proof, rss_bytes=int(rss[1]) if rss else None, heap_peak_bytes=int(heap[1]) if heap else None)

def run_one(root, out, case, side, reps, pair, memory=False):
    # Benchmark metadata retains output paths. Equal-length names avoid counting
    # the spelling of "baseline" versus "candidate" as an allocation regression.
    path_side = 'reference' if memory and side == 'baseline' else side
    path_pair = pair.replace('baseline', 'reference') if memory else pair
    run = out / case['id'] / f'{path_pair}-{path_side}'
    executable_hash = file_hash(root / side / case['exe'])
    request = dict(executable_sha256=executable_hash, case=case, reps=reps, memory=memory)
    if memory:
        request['path_layout'] = 'equal-length-v1'
    if (run / 'result.json').exists():
        cached = json.loads((run / 'result.json').read_text())
        if cached.get('request') != request:
            raise ValueError(f'cached run has different inputs or binary: {run}')
        return cached
    run.mkdir(parents=True, exist_ok=True)
    env = filtered_environment(os.environ, prefixes=('BITZ_', 'FLOCK_', 'RAYON_'))
    env.update(case['env'])
    env.update(RAYON_NUM_THREADS=str(case['threads']), BITZ_BENCH_REPS=str(reps), BITZ_BENCH_SEED='0x5533326d756c0064', BITZ_MULTISWAP_TRACE_PATH=str(run / 'trace.jsonl'))
    command = [str(root / path_side / case['exe']), *case['args']]
    if case['exe'] == 'sha256_ecdsa':
        command.append(str(reps))
    elif case['exe'] == 'mul_bitz':
        command += ['--reps', str(reps), '--threads', str(case['threads']), '--out', str(run / 'mul')]
        if memory:
            command += ['--memory', 'heap']
    started = time.time()
    print(f"RUN {case['id']} {pair} {side}", flush=True)
    with open(run / 'stdout.log', 'w') as stdout, open(run / 'stderr.log', 'w') as stderr:
        exit_code, _ = run_process(['/usr/bin/time', '-l', *command], env=env, stdout=stdout, stderr=stderr)
    data = dict(request=request, command=command, environment={k: v for k, v in env.items() if k.startswith(('BITZ_', 'RAYON_', 'PERFETTO_'))}, exit_code=exit_code, wall_seconds=time.time() - started)
    if exit_code == 0:
        try:
            data.update(parse_run(case, run))
            assert (memory and case['exe'] == 'mul_bitz') or len(data['samples']) == reps, 'measured sample count differs from requested repetitions'
            assert all(math.isfinite(s[k]) and s[k] > 0 for s in data['samples'] for k in ('prove_ms', 'verify_ms')), 'invalid prover/verifier timing'
            data['status'] = 'ok'
        except Exception as e:
            data.update(status='parse_error', error=repr(e))
    else:
        data['status'] = 'failed'
    write_json(run / 'result.json', data)
    print(f"DONE {case['id']} {pair} {side} {data['status']} {data['wall_seconds']:.1f}s", flush=True)
    return data

def compare(pairs, metric):
    if any((a.get('status') != 'ok' or b.get('status') != 'ok' for a, b in pairs)):
        return dict(status='failed_run')
    ratios = []
    baseline_medians = []
    candidate_medians = []
    for a, b in pairs:
        av = statistics.median((s[metric] for s in a['samples']))
        bv = statistics.median((s[metric] for s in b['samples']))
        baseline_medians.append(av)
        candidate_medians.append(bv)
        ratios.append(math.log(bv / av))
    medians = dict(baseline_median=statistics.median(baseline_medians), candidate_median=statistics.median(candidate_medians))
    if len(ratios) < 2:
        return dict(**medians, ratio=math.exp(ratios[0]), status='inconclusive')
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(len(ratios), 2.262)
    mean = statistics.mean(ratios)
    radius = t * statistics.stdev(ratios) / math.sqrt(len(ratios))
    lo, hi = (math.exp(mean - radius), math.exp(mean + radius))
    return dict(**medians, ratio=math.exp(mean), ci95=[lo, hi], status='pass' if hi <= 1.01 else 'regression' if lo > 1.01 else 'inconclusive', paired_log_ratios=ratios)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pairs', type=int, choices=range(1, 11), default=5)
    parser.add_argument('--reps', type=int, default=32)
    parser.add_argument('--filter', default='')
    parser.add_argument('--case', action='append', help='exact case ID; repeat to select several cases')
    parser.add_argument('--aa', action='store_true')
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    assert platform.system() == 'Darwin' and platform.machine() == 'arm64', 'Apple Silicon campaign only'
    if args.reps < 1:
        parser.error('--reps must be positive')
    root = args.artifacts.resolve()
    if args.memory:
        reference = root / 'reference'
        if not reference.exists():
            reference.symlink_to('baseline', target_is_directory=True)
        assert reference.resolve() == (root / 'baseline').resolve(), 'unexpected reference alias'
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    available = cases(args.extended)
    if args.case and set(args.case) - {c['id'] for c in available}:
        parser.error('--case contains an unknown case ID')
    selected = [c for c in available if args.filter in c['id'] and (not args.case or c['id'] in args.case)]
    (out / 'matrix.json').write_text(json.dumps(selected, indent=2))
    summary = []
    if not selected:
        parser.error('--filter matched no cells')
    for case in selected:
        paired = []
        for pair in range(args.pairs):
            order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
            results = {}
            for side in order:
                source = 'baseline' if args.aa else side
                data = run_one(root, out, case, source, args.reps, f'{pair}-{side}', args.memory)
                results[side] = data
            paired.append((results['baseline'], results['candidate']))
        row = dict(case=case['id'])
        if args.memory:
            measurements = [(a.get('heap_peak_bytes'), b.get('heap_peak_bytes')) for a, b in paired]
            row['heap_bytes'] = measurements
            row['memory_gate'] = 'missing' if any((a is None or b is None for a, b in measurements)) else 'pass' if all((b <= a for a, b in measurements)) else 'regression'
        else:
            row.update(prover=compare(paired, 'prove_ms'), verifier=compare(paired, 'verify_ms'))
            if all((a.get('status') == b.get('status') == 'ok' for a, b in paired)):
                keys = set.intersection(*(set(s) for pair in paired for r in pair for s in r['samples']))
                all_keys = set.union(*(set(s) for pair in paired for r in pair for s in r['samples']))
                row['incomplete_phase_coverage'] = sorted(k for k in all_keys - keys if k.startswith('phase:'))
                details = [k for k in keys if k.startswith('phase:') and any((x in k for x in ('inner', 'outer', 'bind', 'piop', 'step', 'mc:presum', 'mc:forest', 'mq:lig')))]
                row['phases'] = {k: compare(paired, k) for k in sorted(details) if all((s[k] > 0 for pair in paired for r in pair for s in r['samples']))}
                row['regressed_phases'] = [k for k, v in row['phases'].items() if v['status'] == 'regression']
        row['proof_size_equal'] = None if args.memory and case['exe'] == 'mul_bitz' else all((a.get('status') == b.get('status') == 'ok' and bool(a.get('proof_bytes')) and (a.get('proof_bytes') == b.get('proof_bytes')) for a, b in paired))
        summary.append(row)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(row), flush=True)
if __name__ == '__main__':
    main()
