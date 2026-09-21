"""Aggregate LeWM++ result JSON files without pooling experiment blocks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TASKS = ('cube', 'pusht', 'reacher', 'tworoom')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for path in Path(args.results_root).rglob('result.json'):
        result = json.loads(path.read_text())
        if result.get('suite') != 'lewm_4tasks':
            continue
        protocol = result['protocol']
        components = result['components']
        representation_mode = components.get('action_prior_representation_mode')
        if representation_mode is None and components['action_prior_mode'] != 'zero':
            representation_mode = 'unknown'
        rows.append(
            {
                'group': result['experiment_group'],
                'family': result['generator_family'],
                'generator_type': result.get('generator_type') or 'no_generator',
                'action_prior_mode': components['action_prior_mode'],
                'action_prior_representation_mode': representation_mode or 'no_action_prior',
                'horizon': int(protocol['goal_offset_steps']),
                'variant': result['variant'],
                'seed': int(protocol['seed']),
                'task': result['task'],
                'success_rate': float(result['success_rate']),
            }
        )
    if not rows:
        raise SystemExit(f'No result.json files found under {args.results_root}')

    grouped = defaultdict(list)
    for row in rows:
        key = (
            row['group'],
            row['family'],
            row['generator_type'],
            row['action_prior_mode'],
            row['action_prior_representation_mode'],
            row['horizon'],
            row['variant'],
        )
        grouped[key].append(row)

    output_rows = []
    for key, group_rows in sorted(grouped.items()):
        seeds = sorted({row['seed'] for row in group_rows})
        by_seed_task = {(row['seed'], row['task']): row['success_rate'] for row in group_rows}
        missing = [(seed, task) for seed in seeds for task in TASKS if (seed, task) not in by_seed_task]
        if missing:
            raise ValueError(f'Incomplete group {key}; missing {missing}')
        record = dict(
            zip(
                (
                    'group',
                    'family',
                    'generator_type',
                    'action_prior_mode',
                    'action_prior_representation_mode',
                    'horizon',
                    'variant',
                ),
                key,
            )
        )
        for task in TASKS:
            values = np.asarray([by_seed_task[seed, task] for seed in seeds])
            record[f'{task}_mean'] = float(values.mean())
            record[f'{task}_std'] = float(values.std(ddof=0))
        macro = np.asarray([np.mean([by_seed_task[seed, task] for task in TASKS]) for seed in seeds])
        record['macro_mean'] = float(macro.mean())
        record['macro_std'] = float(macro.std(ddof=0))
        record['seeds'] = ' '.join(map(str, seeds))
        output_rows.append(record)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    print(f'Wrote {len(output_rows)} aggregate rows to {output}')


if __name__ == '__main__':
    main()
