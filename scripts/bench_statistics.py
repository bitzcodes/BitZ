"""Shared sample statistics and paired performance uncertainty."""
import math
import random
import statistics

def paired_interval(ratios):
    if len(ratios) < 2 or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError('paired intervals require at least two positive finite ratios')
    rng = random.Random(0)
    logs = [math.log(x) for x in ratios]
    draws = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(10000))
    return [percentile(draws, .025), percentile(draws, .975)]


def classify_interval(interval, maximum=1.0):
    """Uncertainty is not evidence of nonregression. No tolerated slowdown."""
    if interval[1] <= maximum:
        return 'pass'
    if interval[0] > 1.0:
        return 'regression'
    return 'inconclusive'


def timing_ratios(baseline, candidate, diagnostic=False):
    if len(baseline) != len(candidate):
        raise ValueError('unpaired timing blocks')
    if any(not math.isfinite(v) or v < 0 for v in baseline + candidate):
        raise ValueError('invalid timing')
    if any(v == 0 for v in baseline + candidate):
        if diagnostic:
            return []  # A removed optional phase has no log-ratio interval.
        raise ValueError('missing mandatory timing')
    return [b/a for a,b in zip(baseline, candidate)]



def percentile(values, probability):
    """Type-7 percentile, including the endpoints."""
    if not values or not 0 <= probability <= 1:
        raise ValueError("percentile requires samples and a probability in [0, 1]")
    values = sorted(values)
    position = (len(values) - 1) * probability
    index = math.floor(position)
    fraction = position - index
    return values[index] + fraction * (values[min(index + 1, len(values) - 1)] - values[index])


def sample_statistics(values):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("expected finite, nonempty samples")
    return dict(median=statistics.median(values), p05=percentile(values,.05), p95=percentile(values,.95),
                minimum=min(values), maximum=max(values), count=len(values))
