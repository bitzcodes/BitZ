#!/usr/bin/env python3
"""Apply the fixed 24-block GKR gates; missing coverage never passes.

Inputs are summary.json files from fresh confirmation runs, not tuning runs.
The intervals are per case; this does not claim universal nonregression.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

from bench_statistics import paired_interval, classify_interval


def evaluate(ecdsa, workloads):
    expected_ecdsa = set(itertools.product(['bitz-split', 'bitz-all'], [7, 10, 12], [1, 10], [100, 128], [31, 47]))
    cases = ([f'{w}-n{n}' for w in ['u32-mod32', 'u64', 'u128'] for n in [15, 19]]
             + [f'full-u32-w{w}-n{n}' for w in [1, 8] for n in [15, 19]]
             + [f'sha-n{n}' for n in [7, 10, 12]]
             + [f'multiswap-b{b}' for b in [1, 2, 4, 8]])
    expected_workloads = set(itertools.product(cases, [1, 10], [31, 47]))
    seen_e, seen_w = set(), set()
    gates = []
    primary = {}
    for kind, rows in [('ecdsa', ecdsa), ('workload', workloads)]:
        for row in rows:
            if kind == 'ecdsa':
                key = tuple(row[k] for k in ['method', 'exponent', 'threads', 'target', 'seed'])
                seen = seen_e
                total = 'e2e_prover_ms'
                if key[:4] == ('bitz-split', 10, 10, 100):
                    primary[key[4]] = row['metrics']['gkr_ms']['paired_ratios']
            else:
                key = tuple(row[k] for k in ['case', 'threads', 'seed'])
                seen = seen_w
                total = 'prove_ms' if row['case'].startswith('full-u32-') else 'e2e_ms'
                if row.get('total_metric', total) != total:
                    raise ValueError('incorrect proving boundary')
            if key in seen:
                raise ValueError(f'duplicate case {key}')
            seen.add(key)
            for metric in [total, 'prove_ms', 'cold_' + total, 'cold_prove_ms', 'cold_gkr_ms']:
                ratios = row['metrics'][metric]['paired_ratios']
                if len(ratios) != 24:
                    raise ValueError(f'{key}: confirmation requires exactly 24 blocks')
                interval = paired_interval(ratios)
                gates.append(dict(case=key, metric=metric, interval=interval, result=classify_interval(interval)))
    missing = sorted(expected_ecdsa - seen_e) + sorted(expected_workloads - seen_w)
    if set(primary) == {31, 47}:
        if any(len(r) != 24 for r in primary.values()):
            raise ValueError('primary confirmation requires 24 blocks per seed')
        combined = [math.sqrt(a*b) for a,b in zip(primary[31], primary[47])]
        interval = paired_interval(combined)
        gates.append(dict(case='primary, equal seed weights', metric='warm_gkr_2pct', interval=interval,
                          result=classify_interval(interval, 0.98)))
    else:
        missing.append('primary GKR confirmation for seeds 31 and 47')
    return dict(passed=not missing and all(g['result'] == 'pass' for g in gates), missing=missing, gates=gates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ecdsa', type=Path, nargs='+', required=True)
    parser.add_argument('--workloads', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    load = lambda paths: [row for path in paths for row in json.loads(path.read_text())]
    result = evaluate(load(args.ecdsa), load(args.workloads))
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
