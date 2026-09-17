"""Evaluate LeWM++, its ablations, LeWM, and Action-Prior-Chunk on LeWM-4Tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from action_prior_runtime_lewm_control import FinalGoalPolicy, load_action_prior
from agents.action_prior_chunk import (
    REPRESENTATION_MODES,
    resolve_representation_mode,
    validate_representation_sharing,
)
from latent_path_flow_lewm_control import LATENT_PATH_FLOW_ARCHITECTURE
from lewm_jax.planner_lewm_control import LeWMPPController

from ogbench.lewm_envs.evaluation import (
    HDF5EvaluationDataset,
    StandardActionScaler,
    evaluate_dataset_goals,
    json_safe,
    task_paths,
)

VARIANTS = ('full', 'no_subgoal', 'no_action_prior', 'no_moh', 'lewm', 'action_prior_chunk')
DEFAULT_CEM_SAMPLES = 300
DEFAULT_CEM_ITERATIONS = 5
DEFAULT_FLOW_STEPS = 16


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('cube', 'pusht', 'reacher', 'tworoom'), required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--experiment-group', required=True)
    parser.add_argument(
        '--generator-family',
        choices=('goalmax25', 'general_uniform_future', 'no_generator'),
        required=True,
    )
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--lewm-checkpoint', required=True)
    parser.add_argument('--action-prior-checkpoint-dir')
    parser.add_argument('--action-prior-checkpoint-step', type=int, default=100_000)
    parser.add_argument(
        '--action-prior-mode',
        choices=('zero', 'policy_mode', 'policy_mode_anchor'),
        default='policy_mode',
    )
    parser.add_argument(
        '--action-prior-representation-mode',
        choices=tuple(REPRESENTATION_MODES),
        default='all',
    )
    parser.add_argument('--subgoal-generator-checkpoint')
    parser.add_argument('--flow-sampling-steps', type=int, default=DEFAULT_FLOW_STEPS)
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--goal-offset-steps', type=int, default=25)
    parser.add_argument('--eval-budget', type=int, default=50)
    parser.add_argument('--cem-horizon', type=int, default=2)
    parser.add_argument('--cem-receding-horizon', type=int, default=1)
    parser.add_argument('--action-block', type=int, default=5)
    parser.add_argument('--cem-num-samples', type=int, default=DEFAULT_CEM_SAMPLES)
    parser.add_argument('--cem-iterations', type=int, default=DEFAULT_CEM_ITERATIONS)
    parser.add_argument('--cem-topk', type=int, default=30)
    parser.add_argument('--cem-var-scale', type=float, default=1.0)
    parser.add_argument('--cem-cost-mode', choices=('last', 'moh'), default='moh')
    parser.add_argument('--video-dir')
    parser.add_argument('--output', required=True)
    parser.add_argument('--validate-only', action='store_true')
    return parser.parse_args()


def expected_components(variant):
    """Return (subgoal, action prior, CEM cost, direct policy)."""
    return {
        'full': (True, True, 'moh', False),
        'no_subgoal': (False, True, 'moh', False),
        'no_action_prior': (True, False, 'moh', False),
        'no_moh': (True, True, 'last', False),
        'lewm': (False, False, 'last', False),
        'action_prior_chunk': (False, True, None, True),
    }[variant]


def validate_args(args):
    for name in (
        'action_prior_checkpoint_step',
        'flow_sampling_steps',
        'num_eval',
        'goal_offset_steps',
        'eval_budget',
        'cem_horizon',
        'cem_receding_horizon',
        'action_block',
        'cem_num_samples',
        'cem_iterations',
        'cem_topk',
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive.')
    if args.cem_var_scale <= 0:
        raise ValueError('CEM variance scale must be positive.')

    use_subgoal, use_prior, cost_mode, direct_policy = expected_components(args.variant)
    if use_subgoal != (args.subgoal_generator_checkpoint is not None):
        raise ValueError(f'Variant {args.variant} has an invalid subgoal-generator setting.')
    if use_prior != (args.action_prior_checkpoint_dir is not None):
        raise ValueError(f'Variant {args.variant} has an invalid action-prior checkpoint setting.')
    if use_prior != (args.action_prior_mode != 'zero'):
        raise ValueError(f'Variant {args.variant} has an invalid action-prior mode.')
    if direct_policy and args.action_prior_mode != 'policy_mode':
        raise ValueError('Direct Action-Prior-Chunk evaluation requires action_prior_mode=policy_mode.')
    if not direct_policy and args.cem_cost_mode != cost_mode:
        raise ValueError(f'Variant {args.variant} requires CEM cost {cost_mode}.')
    if use_subgoal == (args.generator_family == 'no_generator'):
        raise ValueError('Generator-family label does not match the selected variant.')


def require_file(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}')
    return path


def require_path(path, label):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'{label} not found: {path}')
    return path


def sha256_file(path, block_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_release_files(args):
    """Validate all data and checkpoint metadata before allocating an environment."""
    use_subgoal, use_prior, _, _ = expected_components(args.variant)
    lewm_checkpoint = require_file(args.lewm_checkpoint, 'LeWM checkpoint')
    hdf5_path, lance_path = task_paths(args.task, args.data_root)
    require_file(hdf5_path, 'evaluation HDF5 dataset')

    if use_prior:
        require_path(lance_path, 'action-prior Lance dataset')
        checkpoint_dir = Path(args.action_prior_checkpoint_dir).expanduser().resolve()
        flags_path = require_file(checkpoint_dir / 'flags.json', 'action-prior flags')
        flags = json.loads(flags_path.read_text())
        saved_dataset = Path(flags.get('dataset_path') or '')
        if args.task not in saved_dataset.stem.lower():
            raise ValueError(f'Action-prior dataset does not match task {args.task}: {saved_dataset}')
        if flags.get('seed') != 777:
            raise ValueError(f'Action-prior training seed must be 777, got {flags.get("seed")!r}.')
        if flags.get('agent', {}).get('chunk_size') != args.action_block:
            raise ValueError('Action-prior chunk size does not match --action-block.')
        representation = flags.get('representation', {})
        representation_mode = resolve_representation_mode(representation, flags.get('agent', {}))
        if representation_mode != args.action_prior_representation_mode:
            raise ValueError(
                f'Action-prior checkpoint uses representation mode {representation_mode!r}, '
                f'but evaluation requested {args.action_prior_representation_mode!r}.'
            )
        expected_sharing = validate_representation_sharing(
            representation_mode,
            flags.get('agent', {}),
            label='metadata',
        )
        for module, shared in expected_sharing.items():
            recorded_source = representation.get(module)
            expected_source = 'lewm' if shared else 'pixel'
            if recorded_source not in (None, expected_source):
                raise ValueError(
                    f'Action-prior representation metadata is inconsistent: {module}={recorded_source!r}, '
                    f'expected {expected_source!r} for mode={representation_mode!r}.'
                )
        expected_lewm_sha = flags.get('lewm_checkpoint_sha256')
        if expected_lewm_sha is not None and sha256_file(lewm_checkpoint) != expected_lewm_sha:
            raise ValueError('Action prior and evaluator use different frozen LeWM checkpoints.')
        require_file(
            checkpoint_dir / f'params_{args.action_prior_checkpoint_step}.pkl',
            'action-prior checkpoint',
        )

    if not use_subgoal:
        return

    checkpoint = require_file(args.subgoal_generator_checkpoint, 'subgoal-generator checkpoint')
    config_path = require_file(checkpoint.parent / 'config.json', 'adjacent generator config')
    config = json.loads(config_path.read_text())

    config_task = config.get('task')
    latent_dataset = Path(config.get('latent_dataset') or '')
    if config_task not in (None, args.task):
        raise ValueError(f'Generator task mismatch: {config_task!r} != {args.task!r}.')
    if config_task is None and args.task not in latent_dataset.stem.lower():
        raise ValueError(f'Generator latent dataset does not match task {args.task}: {latent_dataset}')
    expected_lewm_sha = config.get('lewm_checkpoint_sha256')
    if expected_lewm_sha is None:
        raise ValueError('Generator config does not record lewm_checkpoint_sha256.')
    if sha256_file(lewm_checkpoint) != expected_lewm_sha:
        raise ValueError('Subgoal generator and evaluator use different frozen LeWM checkpoints.')

    if args.goal_offset_steps == 25:
        expected_family = 'goalmax25'
        expected_sampling = 'uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25'
        expected_max_goal_steps = 25
    elif args.goal_offset_steps > 25:
        expected_family = 'general_uniform_future'
        expected_sampling = 'hiql_uniform_future_same_trajectory'
        expected_max_goal_steps = None
    else:
        raise ValueError('The release protocol supports H25 and H>25 only.')

    if args.generator_family != expected_family:
        raise ValueError(f'H{args.goal_offset_steps} requires {expected_family}, got {args.generator_family}.')
    if config.get('goal_sampling') != expected_sampling:
        raise ValueError(
            f'{expected_family} goal_sampling mismatch: {config.get("goal_sampling")!r} != {expected_sampling!r}'
        )
    if config.get('max_goal_steps') != expected_max_goal_steps:
        raise ValueError(
            f'{expected_family} max_goal_steps mismatch: '
            f'{config.get("max_goal_steps")!r} != {expected_max_goal_steps!r}'
        )
    if config.get('architecture') != LATENT_PATH_FLOW_ARCHITECTURE:
        raise ValueError(
            f'LatentPathFlow architecture mismatch: '
            f'{config.get("architecture")!r} != {LATENT_PATH_FLOW_ARCHITECTURE!r}'
        )


def main():
    args = parse_args()
    validate_args(args)
    validate_release_files(args)
    if args.validate_only:
        print(f'Validated {args.variant} task={args.task} H={args.goal_offset_steps} family={args.generator_family}.')
        return
    use_subgoal, use_prior, _, direct_policy = expected_components(args.variant)
    hdf5_path, lance_path = task_paths(args.task, args.data_root)
    dataset = HDF5EvaluationDataset(hdf5_path)
    try:
        episodes, starts = dataset.sample_starts(args.num_eval, args.goal_offset_steps, args.seed)
        scaler = StandardActionScaler(dataset.get_column('action'))
        action_prior = (
            load_action_prior(
                lance_path,
                args.action_prior_checkpoint_dir,
                args.action_prior_checkpoint_step,
                args.lewm_checkpoint,
                args.action_prior_representation_mode,
            )
            if use_prior
            else None
        )
        if direct_policy:
            controller = FinalGoalPolicy(action_prior, scaler, args.seed)
        else:
            controller = LeWMPPController(
                checkpoint=args.lewm_checkpoint,
                scaler=scaler,
                seed=args.seed,
                horizon=args.cem_horizon,
                receding_horizon=args.cem_receding_horizon,
                action_block=args.action_block,
                num_samples=args.cem_num_samples,
                iterations=args.cem_iterations,
                topk=args.cem_topk,
                var_scale=args.cem_var_scale,
                cost_mode=args.cem_cost_mode,
                action_prior=action_prior,
                action_prior_mode=args.action_prior_mode,
                paired_plan_keys=True,
                subgoal_generator_checkpoint=args.subgoal_generator_checkpoint,
                flow_sampling_steps=args.flow_sampling_steps,
            )
            if use_subgoal:
                actual = controller.subgoal_generator.config['architecture']
                if actual != LATENT_PATH_FLOW_ARCHITECTURE:
                    raise ValueError(
                        f'Generator type mismatch: checkpoint has {actual!r}, '
                        f'expected {LATENT_PATH_FLOW_ARCHITECTURE!r}.'
                    )

        started = time.time()
        metrics = evaluate_dataset_goals(
            task=args.task,
            dataset=dataset,
            episodes=episodes,
            starts=starts,
            goal_offset=args.goal_offset_steps,
            eval_budget=args.eval_budget,
            policy=controller,
            video_dir=args.video_dir,
        )
    finally:
        dataset.close()

    generator = None if direct_policy else controller.subgoal_generator
    result = {
        'suite': 'lewm_4tasks',
        'task': args.task,
        'variant': args.variant,
        'experiment_group': args.experiment_group,
        'generator_family': args.generator_family,
        'generator_type': 'latent_path_flow' if use_subgoal else None,
        'components': {
            'subgoal_generator': use_subgoal,
            'action_prior_mode': args.action_prior_mode,
            'action_prior_representation_mode': action_prior.representation_mode if use_prior else None,
            'min_over_horizon': None if direct_policy else args.cem_cost_mode == 'moh',
            'direct_policy': direct_policy,
        },
        'lewm_checkpoint': str(Path(args.lewm_checkpoint).expanduser().resolve()),
        'action_prior_checkpoint_dir': args.action_prior_checkpoint_dir,
        'action_prior_checkpoint_step': args.action_prior_checkpoint_step if use_prior else None,
        'subgoal_generator': (
            None
            if generator is None
            else {
                'checkpoint': generator.checkpoint,
                'checkpoint_step': generator.checkpoint_step,
                'num_samples': 1,
                'flow_sampling_steps': generator.flow_sampling_steps,
                'config': generator.config,
                'generation_counts': generator.generation_counts,
            }
        ),
        'protocol': {
            'num_eval': args.num_eval,
            'seed': args.seed,
            'goal_offset_steps': args.goal_offset_steps,
            'eval_budget': args.eval_budget,
            'cem_horizon': None if direct_policy else controller.horizon,
            'cem_receding_horizon': None if direct_policy else args.cem_receding_horizon,
            'action_block': args.action_block,
            'cem_num_samples': None if direct_policy else args.cem_num_samples,
            'cem_iterations': None if direct_policy else args.cem_iterations,
            'cem_topk': None if direct_policy else args.cem_topk,
            'cem_var_scale': None if direct_policy else args.cem_var_scale,
            'cem_cost_mode': None if direct_policy else args.cem_cost_mode,
            'flow_sampling_steps': args.flow_sampling_steps if use_subgoal else None,
            'policy_goal': 'final_goal',
            'paired_plan_keys': not direct_policy,
        },
        'metrics': metrics,
        'success_rate': metrics['success_rate'],
        'evaluation_time': time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), indent=2) + '\n')
    print(json.dumps(json_safe(result), indent=2))


if __name__ == '__main__':
    main()
