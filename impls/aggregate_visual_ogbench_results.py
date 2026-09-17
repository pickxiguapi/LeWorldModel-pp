"""Aggregate the exact 8 datasets x 3 evaluation-seed paper matrix."""

import argparse
import json
import statistics
from pathlib import Path

TAGS = (
    'cs_play', 'cd_play', 'ct_play', 'scene_play',
    'cs_noisy', 'cd_noisy', 'ct_noisy', 'scene_noisy',
)
SEEDS = (0, 1, 42)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', choices=('lewmpp', 'lewm'), default='lewmpp')
    args = parser.parse_args()
    root = Path(args.results_root)
    values = {tag: [] for tag in TAGS}
    for seed in SEEDS:
        for tag in TAGS:
            path = root / tag / f'seed{seed}' / 'result.json'
            if not path.is_file():
                raise SystemExit(f'Incomplete matrix; missing {path}')
            result = json.loads(path.read_text())
            if result.get('seed') != seed or result.get('episodes_per_task') != 50:
                raise SystemExit(f'Protocol mismatch in {path}')
            if args.method == 'lewmpp':
                valid_controller = result.get('use_subgoal') and result.get('policy_guidance') == 'policy_random_mixture'
            else:
                valid_controller = not result.get('use_subgoal') and result.get('policy_guidance') == 'none'
            if not valid_controller:
                raise SystemExit(f'Controller mismatch for {args.method} in {path}')
            values[tag].append(float(result['overall_success']))
    rows = {
        tag: {
            'per_seed': dict(zip(map(str, SEEDS), rates)),
            'mean': statistics.mean(rates),
            'sample_std': statistics.stdev(rates),
        }
        for tag, rates in values.items()
    }
    macro_per_seed = [statistics.mean(values[tag][i] for tag in TAGS) for i in range(len(SEEDS))]
    summary = {
        'evaluation_seeds': list(SEEDS),
        'method': args.method,
        'episodes_per_official_task': 50,
        'std_definition': 'sample standard deviation across evaluation seeds (ddof=1)',
        'by_environment': rows,
        'macro': {
            'per_seed': dict(zip(map(str, SEEDS), macro_per_seed)),
            'mean': statistics.mean(macro_per_seed),
            'sample_std': statistics.stdev(macro_per_seed),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + '\n')
    for tag, row in rows.items():
        print(f'{tag}: {100 * row["mean"]:.2f} ± {100 * row["sample_std"]:.2f}')
    print(f'macro: {100 * summary["macro"]["mean"]:.2f} ± {100 * summary["macro"]["sample_std"]:.2f}')


if __name__ == '__main__':
    main()
