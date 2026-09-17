"""Evaluate LeWM and LeWM++ on the eight Visual OGBench datasets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from action_prior_runtime_ogbench import LeWMEncodedAgent, load_agent_config
from lewm_jax import load_frozen_lewm
from lewm_jax.planner_ogbench import JAXLeWMCEMPolicy
from tqdm import trange

import ogbench
from ogbench.lewm_envs.evaluation import json_safe


class NPZActionScaler:
    """Match the action normalization used by LeWMNPZSequenceDataset."""

    def __init__(self, dataset_path):
        with np.load(dataset_path) as archive:
            actions = archive['actions']
            terminals = archive['terminals'].astype(bool, copy=False)
        actions = actions[~terminals]
        actions = actions[~np.isnan(actions).any(axis=1)]
        self.mean = actions.mean(axis=0)
        self.scale = actions.std(axis=0, ddof=1)
        self.scale = np.where(self.scale > 0, self.scale, 1.0)
        self.action_dim = int(actions.shape[-1])

    def inverse_transform(self, value):
        return np.asarray(value) * self.scale + self.mean

    def transform(self, value):
        return (np.asarray(value) - self.mean) / self.scale


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-name', required=True)
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument(
        '--policy-guidance',
        choices=('none', 'policy_random_mixture'),
        default='none',
    )
    parser.add_argument('--guidance-population-size', type=int, default=0)
    parser.add_argument('--guidance-temperature', type=float, default=1.0)
    parser.add_argument('--guidance-first-block-std', type=float)
    parser.add_argument('--guidance-random-elite-cap', type=int, default=0)
    parser.add_argument('--guidance-mean-residual-weight', type=float, default=1.0)
    parser.add_argument('--use-subgoal', action='store_true')
    parser.add_argument('--lewm-checkpoint', required=True)
    parser.add_argument('--policy-checkpoint-dir')
    parser.add_argument('--policy-checkpoint-step', type=int, default=500_000)
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cem-horizon', type=int, default=5)
    parser.add_argument('--cem-receding-horizon', type=int, default=1)
    parser.add_argument('--action-block', type=int, default=5)
    parser.add_argument('--cem-num-samples', type=int, default=300)
    parser.add_argument('--cem-iterations', type=int, default=30)
    parser.add_argument('--cem-topk', type=int, default=30)
    parser.add_argument('--cem-var-scale', type=float, default=1.0)
    parser.add_argument('--cem-cost-mode', choices=('last', 'moh'), default='moh')
    parser.add_argument('--latent-subgoal-checkpoint')
    parser.add_argument('--video-dir')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def load_policy(env, checkpoint_dir, checkpoint_step):
    from agents.action_prior_chunk_lewm_ogbench import LeWMGCIQLChunkAgent
    from agents.action_prior_chunk_ogbench import GCIQLChunkAgent
    from utils.flax_utils import restore_agent

    name, config, saved = load_agent_config(checkpoint_dir)
    observation = np.zeros(
        (1, *env.observation_space.shape), dtype=env.observation_space.dtype
    )
    action_width = int(np.prod(env.action_space.shape)) * int(config.chunk_size)
    actions = np.zeros((1, action_width), dtype=np.float32)
    if name == 'gciql_chunk':
        agent = GCIQLChunkAgent.create(0, observation, actions, config)
        return restore_agent(agent, checkpoint_dir, checkpoint_step), saved

    lewm_checkpoint = saved.get('lewm_checkpoint')
    if lewm_checkpoint is None:
        lewm_checkpoint = saved['representation']['lewm_checkpoint']
    model, variables, metadata = load_frozen_lewm(lewm_checkpoint)
    agent = LeWMGCIQLChunkAgent.create(
        0,
        observation,
        np.zeros((1, int(metadata['config']['embed_dim'])), dtype=np.float32),
        actions,
        config,
    )
    agent = restore_agent(agent, checkpoint_dir, checkpoint_step)
    encode_pixels = jax.jit(
        lambda pixels: model.apply(
            variables, pixels, train=False, method=model.encode_pixels
        ).astype(jnp.float32)
    )
    return (
        LeWMEncodedAgent(
            agent,
            encode_pixels,
            share_pi_encoder=config.share_pi_encoder,
            lewm_checkpoint=metadata['path'],
        ),
        saved,
    )


def main():
    args = parse_args()
    needs_policy = args.policy_guidance != 'none'
    needs_subgoal = args.use_subgoal
    if needs_policy != (args.policy_checkpoint_dir is not None):
        raise ValueError('Invalid guidance/policy-checkpoint combination.')
    if needs_subgoal != (args.latent_subgoal_checkpoint is not None):
        raise ValueError(
            'Invalid use-subgoal/--latent-subgoal-checkpoint combination.'
        )
    if args.guidance_population_size < 0:
        raise ValueError('--guidance-population-size must be non-negative.')
    if args.guidance_temperature < 0:
        raise ValueError('--guidance-temperature must be non-negative.')
    if (
        args.guidance_first_block_std is not None
        and args.guidance_first_block_std <= 0
    ):
        raise ValueError('--guidance-first-block-std must be positive.')
    if args.guidance_random_elite_cap < 0:
        raise ValueError('--guidance-random-elite-cap must be non-negative.')
    if not 0.0 <= args.guidance_mean_residual_weight <= 1.0:
        raise ValueError('--guidance-mean-residual-weight must be in [0, 1].')

    np.random.seed(args.seed)
    env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    env.reset(seed=args.seed)
    scaler = NPZActionScaler(args.dataset_path)
    policy_agent = None
    representation_mode = None
    if needs_policy:
        policy_agent, policy_flags = load_policy(
            env,
            args.policy_checkpoint_dir,
            args.policy_checkpoint_step,
        )
        representation_mode = policy_flags.get('representation', {}).get(
            'mode', 'independent'
        )
    policy = JAXLeWMCEMPolicy(
        args.lewm_checkpoint,
        scaler,
        seed=args.seed,
        horizon=args.cem_horizon,
        receding_horizon=args.cem_receding_horizon,
        action_block=args.action_block,
        num_samples=args.cem_num_samples,
        iterations=args.cem_iterations,
        topk=args.cem_topk,
        var_scale=args.cem_var_scale,
        cost_mode=args.cem_cost_mode,
        guidance_policy=policy_agent,
        guidance_mode=args.policy_guidance,
        guidance_population_size=args.guidance_population_size,
        guidance_temperature=args.guidance_temperature,
        guidance_first_block_std=args.guidance_first_block_std,
        guidance_random_elite_cap=args.guidance_random_elite_cap,
        guidance_mean_residual_weight=args.guidance_mean_residual_weight,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        latent_subgoal_checkpoint=args.latent_subgoal_checkpoint,
    )

    task_infos = env.unwrapped.task_infos
    metrics = {}
    all_successes = []
    video_dir = Path(args.video_dir) if args.video_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        for task_id, task_info in enumerate(task_infos, start=1):
            successes = []
            for episode in trange(args.num_eval, desc=task_info['task_name']):
                observation, info = env.reset(options={'task_id': task_id})
                goal = np.asarray(info['goal'], dtype=np.uint8)
                frames = [np.asarray(observation)] if video_dir is not None else None
                policy.reset(env.action_space, num_envs=1)
                done = False
                while not done:
                    action = policy.get_actions(
                        np.asarray(observation, dtype=np.uint8)[None, None],
                        goal[None, None],
                        np.asarray([True]),
                    )[0]
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    observation, _, terminated, truncated, info = env.step(action)
                    if frames is not None:
                        frames.append(np.asarray(observation))
                    done = terminated or truncated
                successes.append(float(info['success']))
                if frames is not None:
                    import imageio.v2 as imageio

                    imageio.mimsave(
                        video_dir / f'{task_id:02d}_{task_info["task_name"]}_ep{episode}.mp4',
                        frames,
                        fps=20,
                    )
            metrics[task_info['task_name']] = float(np.mean(successes))
            all_successes.extend(successes)
    finally:
        env.close()

    result = {
        'suite': 'ogbench_env_8tasks',
        'environment': args.env_name,
        'controller': 'lewm_cem',
        'policy_guidance': args.policy_guidance,
        'policy_guidance_config': {
            'population_size': args.guidance_population_size,
            'random_size': (
                args.cem_num_samples - args.guidance_population_size
                if args.policy_guidance == 'policy_random_mixture'
                else 0
            ),
            'temperature': args.guidance_temperature,
            'first_block_std': args.guidance_first_block_std,
            'random_elite_cap': args.guidance_random_elite_cap,
            'mean_residual_weight': args.guidance_mean_residual_weight,
            'refreshes_policy_population_each_iteration': (
                args.policy_guidance == 'policy_random_mixture'
            ),
            'executes_best_final_candidate': (
                args.policy_guidance == 'policy_random_mixture'
            ),
        },
        'guidance_goal_mode': 'final',
        'use_subgoal': args.use_subgoal,
        'representation_mode': representation_mode,
        'lewm_checkpoint': args.lewm_checkpoint,
        'policy_checkpoint_dir': args.policy_checkpoint_dir,
        'policy_checkpoint_step': args.policy_checkpoint_step if needs_policy else None,
        'policy_action_space': 'environment' if needs_policy else None,
        'latent_subgoal': (
            None
            if not needs_subgoal
            else {
                'checkpoint': policy.latent_subgoal_checkpoint,
                'lewm_checkpoint': policy.lewm_checkpoint,
                'checkpoint_step': policy.latent_subgoal_checkpoint_step,
                'num_samples': 1,
                'sample_selection': 'single_sample',
                'training_subgoal_steps': int(
                    policy.latent_subgoal_config['subgoal_steps']
                ),
                'training_action_block': int(
                    policy.latent_subgoal_config['action_block']
                ),
                'selected_waypoint_index': policy.latent_subgoal_waypoint_index,
                'selected_waypoint_step': policy.latent_subgoal_waypoint_step,
                'history_size': policy.latent_subgoal_history_size,
                'generation_counts': policy.latent_subgoal_generation_counts,
            }
        ),
        'cem': {
            'horizon': policy.horizon,
            'receding_horizon': args.cem_receding_horizon,
            'action_block': args.action_block,
            'num_samples': args.cem_num_samples,
            'iterations': args.cem_iterations,
            'topk': args.cem_topk,
            'var_scale': args.cem_var_scale,
            'cost_mode': args.cem_cost_mode,
        },
        'seed': args.seed,
        'episodes_per_task': args.num_eval,
        'metrics': metrics,
        'overall_success': float(np.mean(all_successes)),
        'evaluation_time': time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), indent=2) + '\n')
    print(json.dumps(json_safe(result), indent=2))


if __name__ == '__main__':
    main()
