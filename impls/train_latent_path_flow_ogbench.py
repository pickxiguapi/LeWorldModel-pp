"""Train a latent subgoal generator on frozen LeWM latent caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from latent_path_flow_ogbench import (
    LATENT_PATH_FLOW_ARCHITECTURE,
    LatentPathFlow,
    latent_path_waypoint_steps,
    sample_conditional_path_flow,
)
from utils.latent_path_flow_dataset_ogbench import (
    build_distance_balanced_transition_tables,
    build_history_indices,
    build_valid_transitions,
    load_latent_cache,
    sample_distance_balanced_future_pairs,
    sample_future_pairs,
    split_episodes,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latent-dataset', required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--train-fraction', type=float, default=0.95)
    parser.add_argument('--subgoal-steps', type=int, default=10)
    parser.add_argument('--train-steps', type=int, default=100_000)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--flow-sampling-steps', type=int, default=16)
    parser.add_argument('--ema-decay', type=float, default=0.9999)
    parser.add_argument('--action-block', type=int, default=5)
    parser.add_argument(
        '--goal-sampling',
        choices=('uniform_future', 'aligned_future'),
        default='uniform_future',
    )
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--ff-dim', type=int, default=2048)
    parser.add_argument('--time-dim', type=int, default=64)
    parser.add_argument('--history-size', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--final-learning-rate', type=float, default=3e-5)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--validation-pairs', type=int, default=50_000)
    parser.add_argument('--validation-manifest')
    parser.add_argument('--eval-batch-size', type=int, default=5000)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--eval-interval', type=int, default=5000)
    parser.add_argument('--checkpoint-interval', type=int, default=25_000)
    parser.add_argument('--max-goal-steps', type=int)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def validate_args(args):
    positive = (
        'subgoal_steps',
        'action_block',
        'train_steps',
        'batch_size',
        'warmup_steps',
        'validation_pairs',
        'eval_batch_size',
        'log_interval',
        'eval_interval',
        'checkpoint_interval',
        'num_heads',
        'flow_sampling_steps',
        'hidden_dim',
        'depth',
        'ff_dim',
        'time_dim',
        'history_size',
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive.')
    if args.warmup_steps >= args.train_steps:
        raise ValueError('warmup_steps must be smaller than train_steps.')
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError('train_fraction must be in (0, 1).')
    if args.learning_rate <= 0 or args.final_learning_rate < 0:
        raise ValueError('Learning rates must be non-negative and the peak must be positive.')
    if args.weight_decay < 0 or args.gradient_clip <= 0:
        raise ValueError('weight_decay must be non-negative and gradient_clip must be positive.')
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError('ema_decay must be in [0, 1).')
    latent_path_waypoint_steps(args.subgoal_steps, args.action_block)
    if args.hidden_dim % args.num_heads:
        raise ValueError('hidden_dim must be divisible by num_heads.')
    if args.time_dim % 2:
        raise ValueError('time_dim must be even.')
    if args.max_goal_steps is not None:
        if args.max_goal_steps <= 0:
            raise ValueError('max_goal_steps must be positive when provided.')
        if args.goal_sampling != 'aligned_future':
            raise ValueError('max_goal_steps requires goal_sampling=aligned_future.')
        if args.max_goal_steps % args.action_block:
            raise ValueError('max_goal_steps must be divisible by action_block.')
    if args.history_size <= 1:
        raise ValueError('LatentPathFlow requires --history-size greater than one.')


class FlowTrainState(train_state.TrainState):
    ema_params: Any


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as file:
        json.dump(json_safe(value), file, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def append_jsonl(path, value):
    with Path(path).open('a') as file:
        file.write(json.dumps(json_safe(value), sort_keys=True) + '\n')
        file.flush()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_validation_manifest(path, cache, val_episodes, args):
    """Load and validate a shared fixed-offset validation manifest."""
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as manifest:
        metadata = json.loads(str(manifest['metadata_json'].item()))
        arrays = {
            name: np.asarray(manifest[name], dtype=np.int32)
            for name in (
                'episode_ids',
                'current_indices',
                'history_indices',
                'goal_indices',
                'waypoint_steps',
                'waypoint_target_indices',
                'endpoint_target_indices',
            )
        }
    expected = {
        'format': 'lewm_latent_subgoal_validation_manifest',
        'lewm_checkpoint_sha256': str(
            cache.metadata.get('checkpoint_sha256', '')
        ),
        'num_rows': len(cache.z),
        'num_episodes': len(cache.episode_offsets),
        'split_seed': args.split_seed,
        'train_fraction': args.train_fraction,
        'num_pairs': args.validation_pairs,
        'history_size': args.history_size,
        'subgoal_steps': args.subgoal_steps,
        'action_block': args.action_block,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(
                f'Validation manifest {name} mismatch: '
                f'{metadata.get(name)!r} != {value!r}'
            )
    count = args.validation_pairs
    if arrays['current_indices'].shape != (count,):
        raise ValueError('Validation manifest current_indices has wrong shape.')
    if arrays['history_indices'].shape != (count, args.history_size):
        raise ValueError('Validation manifest history_indices has wrong shape.')
    if arrays['goal_indices'].shape != (count,):
        raise ValueError('Validation manifest goal_indices has wrong shape.')
    if arrays['endpoint_target_indices'].shape != (count,):
        raise ValueError('Validation manifest endpoint targets have wrong shape.')
    if arrays['waypoint_target_indices'].shape[0] != count:
        raise ValueError('Validation manifest waypoint targets have wrong shape.')
    expected_waypoint_steps = np.arange(
        args.action_block,
        args.subgoal_steps + 1,
        args.action_block,
        dtype=np.int32,
    )
    if not np.array_equal(arrays['waypoint_steps'], expected_waypoint_steps):
        raise ValueError('Validation manifest waypoint steps do not match the run.')
    if arrays['waypoint_target_indices'].shape != (
        count,
        len(expected_waypoint_steps),
    ):
        raise ValueError('Validation manifest waypoint target shape is inconsistent.')
    if not set(np.unique(arrays['episode_ids'])).issubset(set(val_episodes)):
        raise ValueError('Validation manifest contains non-validation episodes.')
    index_arrays = (
        arrays['current_indices'],
        arrays['history_indices'],
        arrays['goal_indices'],
        arrays['waypoint_target_indices'],
        arrays['endpoint_target_indices'],
    )
    if any(
        np.any(values < 0) or np.any(values >= len(cache.z))
        for values in index_arrays
    ):
        raise ValueError('Validation manifest contains out-of-range row indices.')
    goal_offset = int(metadata.get('goal_offset', 0))
    if goal_offset < args.subgoal_steps or not np.all(
        arrays['goal_indices'] - arrays['current_indices'] == goal_offset
    ):
        raise ValueError('Validation manifest does not use one exact valid goal offset.')
    expected_waypoint_targets = (
        arrays['current_indices'][:, None] + expected_waypoint_steps[None]
    )
    if not np.array_equal(
        arrays['waypoint_target_indices'], expected_waypoint_targets
    ):
        raise ValueError('Validation manifest waypoint targets are not exact offsets.')
    if not np.array_equal(
        arrays['endpoint_target_indices'], expected_waypoint_targets[:, -1]
    ):
        raise ValueError('Validation manifest endpoint is not the last waypoint.')
    expected_history = build_history_indices(
        arrays['current_indices'], cache.episode_offsets, args.history_size
    )
    if not np.array_equal(arrays['history_indices'], expected_history):
        raise ValueError('Validation manifest history indices are not episode-safe.')
    if not np.array_equal(
        arrays['endpoint_target_indices'], arrays['waypoint_target_indices'][:, -1]
    ):
        raise ValueError('Validation manifest endpoint does not match final waypoint.')
    return arrays, metadata, sha256_file(path)


def checkpoint_path(output_dir, step):
    return Path(output_dir) / f'checkpoint_{int(step):06d}.msgpack'


def save_checkpoint(state, rng, output_dir):
    step = int(jax.device_get(state.step))
    path = checkpoint_path(output_dir, step)
    payload = {
        'step': step,
        'train_state': flax.serialization.to_state_dict(jax.device_get(state)),
        'rng': np.asarray(jax.device_get(rng)),
    }
    serialized = flax.serialization.msgpack_serialize(payload)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as file:
        file.write(serialized)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    return path


def find_latest_checkpoint(output_dir):
    candidates = []
    for path in Path(output_dir).glob('checkpoint_*.msgpack'):
        match = re.fullmatch(r'checkpoint_(\d+)\.msgpack', path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(None, None))


def restore_checkpoint(state, path):
    payload = flax.serialization.msgpack_restore(Path(path).read_bytes())
    restored = flax.serialization.from_state_dict(state, payload['train_state'])
    rng = jnp.asarray(payload['rng'], dtype=jnp.uint32)
    if int(restored.step) != int(payload['step']):
        raise ValueError(f'Checkpoint step mismatch in {path}.')
    return restored, rng


def make_train_step(
    model,
    learning_rate_schedule,
    batch_size,
    subgoal_steps,
    *,
    action_block=5,
    waypoint_steps=None,
    goal_stride=1,
    goal_steps=(),
    history_size=1,
    ema_decay=0.0,
):
    if waypoint_steps is None:
        waypoint_steps = latent_path_waypoint_steps(subgoal_steps, action_block)
    waypoint_steps = tuple(int(step) for step in waypoint_steps)
    goal_steps = jnp.asarray(goal_steps, dtype=jnp.int32)
    distance_balanced = bool(len(goal_steps))

    @jax.jit
    def train_step(state, rng, z, valid_t, valid_history, final_t):
        rng, index_key, goal_key, noise_key, time_key = jax.random.split(rng, 5)
        if distance_balanced:
            distance_idxs = jax.random.randint(
                goal_key, (batch_size,), minval=0, maxval=len(goal_steps)
            )
            candidate_counts = final_t[distance_idxs]
            positions = jnp.floor(
                jax.random.uniform(index_key, (batch_size,)) * candidate_counts
            ).astype(jnp.int32)
            current_idxs = valid_t[distance_idxs, positions]
            history_idxs = valid_history[distance_idxs, positions]
            goal_idxs = current_idxs + goal_steps[distance_idxs]
        else:
            positions = jax.random.randint(
                index_key, (batch_size,), minval=0, maxval=len(valid_t)
            )
            current_idxs = valid_t[positions]
            history_idxs = valid_history[positions]
            final_idxs = final_t[positions]
            distances = jax.random.uniform(goal_key, (batch_size,))
            future_counts = (final_idxs - current_idxs) // goal_stride
            goal_blocks = 1 + jnp.floor(
                distances * future_counts
            ).astype(jnp.int32)
            goal_idxs = current_idxs + goal_blocks * goal_stride
        target_idxs = jnp.stack(
            [jnp.minimum(current_idxs + step, goal_idxs) for step in waypoint_steps],
            axis=1,
        )
        current_latents = z[history_idxs]
        goal_latents = z[goal_idxs]
        target_latents = z[target_idxs]

        def loss_fn(params):
            noise = jax.random.normal(noise_key, target_latents.shape)
            flow_times = jax.random.uniform(time_key, (batch_size,), minval=0.0, maxval=1.0)
            time_broadcast = flow_times[:, None, None]
            interpolation = (1.0 - time_broadcast) * noise + time_broadcast * target_latents
            target_velocity = target_latents - noise
            predictions = model.apply(
                {'params': params}, interpolation, current_latents, goal_latents, flow_times
            )
            errors = predictions - target_velocity
            loss = jnp.mean(jnp.square(errors))
            endpoint_predictions = interpolation + (1.0 - time_broadcast) * predictions
            metrics = {
                'flow_matching_mse': loss,
                'endpoint_proxy_mse': jnp.mean(jnp.square(endpoint_predictions - target_latents)),
                'velocity_norm': jnp.mean(jnp.linalg.norm(predictions, axis=-1)),
                'target_velocity_norm': jnp.mean(jnp.linalg.norm(target_velocity, axis=-1)),
                'target_norm': jnp.mean(jnp.linalg.norm(target_latents, axis=-1)),
            }
            return loss, metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        ema_params = jax.tree_util.tree_map(
            lambda ema, value: ema_decay * ema + (1.0 - ema_decay) * value,
            state.ema_params,
            state.params,
        )
        state = state.replace(ema_params=ema_params)
        metrics['learning_rate'] = learning_rate_schedule(state.step - 1)
        metrics['gradient_norm'] = optax.global_norm(grads)
        return state, rng, metrics

    return train_step


def make_predict_indices(
    model,
    *,
    flow_sampling_steps=16,
):
    @jax.jit
    def predict_indices(params, z, current_idxs, history_idxs, goal_idxs, rng):
        del current_idxs
        return sample_conditional_path_flow(
            model,
            params,
            z[history_idxs],
            z[goal_idxs],
            rng,
            num_steps=flow_sampling_steps,
        )

    return predict_indices


def mean_or_nan(values, mask):
    return float(values[mask].mean()) if np.any(mask) else float('nan')


def evaluate_validation(
    params,
    *,
    z_device,
    z_host,
    current_idxs,
    history_idxs,
    goal_idxs,
    target_idxs,
    predict_indices,
    batch_size,
    seed,
):
    target_shape = z_host[target_idxs].shape[1:]
    predictions = np.empty((len(current_idxs), *target_shape), dtype=np.float32)
    for start in range(0, len(current_idxs), batch_size):
        stop = min(start + batch_size, len(current_idxs))
        batch_rng = jax.random.fold_in(jax.random.PRNGKey(seed), start)
        predictions[start:stop] = np.asarray(
            jax.device_get(
                predict_indices(
                    params,
                    z_device,
                    jnp.asarray(current_idxs[start:stop]),
                    jnp.asarray(history_idxs[start:stop]),
                    jnp.asarray(goal_idxs[start:stop]),
                    batch_rng,
                )
            )
        )
    targets = z_host[target_idxs]
    squared_errors = np.square(predictions - targets)
    per_example_mse = np.mean(squared_errors, axis=tuple(range(1, squared_errors.ndim)))
    deltas = goal_idxs - current_idxs
    flat_predictions = predictions.reshape(len(predictions), -1)
    flat_targets = targets.reshape(len(targets), -1)
    cosine = np.sum(flat_predictions * flat_targets, axis=-1) / (
        np.linalg.norm(flat_predictions, axis=-1)
        * np.linalg.norm(flat_targets, axis=-1)
        + 1e-8
    )
    metrics = {
        'mse': float(per_example_mse.mean()),
        'mse_near': mean_or_nan(per_example_mse, deltas <= 10),
        'mse_medium': mean_or_nan(per_example_mse, (deltas > 10) & (deltas <= 25)),
        'mse_far': mean_or_nan(per_example_mse, deltas > 25),
        'l2': float(np.linalg.norm(flat_predictions - flat_targets, axis=-1).mean()),
        'cosine_similarity': float(cosine.mean()),
        'prediction_norm': float(np.linalg.norm(flat_predictions, axis=-1).mean()),
        'target_norm': float(np.linalg.norm(flat_targets, axis=-1).mean()),
        'near_fraction': float(np.mean(deltas <= 10)),
        'medium_fraction': float(np.mean((deltas > 10) & (deltas <= 25))),
        'far_fraction': float(np.mean(deltas > 25)),
    }
    if predictions.ndim == 3:
        for waypoint_index in range(predictions.shape[1]):
            waypoint_predictions = predictions[:, waypoint_index]
            waypoint_targets = targets[:, waypoint_index]
            waypoint_mse = np.mean(
                np.square(waypoint_predictions - waypoint_targets), axis=-1
            )
            waypoint_cosine = np.sum(
                waypoint_predictions * waypoint_targets, axis=-1
            ) / (
                np.linalg.norm(waypoint_predictions, axis=-1)
                * np.linalg.norm(waypoint_targets, axis=-1)
                + 1e-8
            )
            metrics[f'waypoint_{waypoint_index}_mse'] = float(waypoint_mse.mean())
            metrics[f'waypoint_{waypoint_index}_cosine_similarity'] = float(
                waypoint_cosine.mean()
            )
    return metrics


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.save_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / 'metrics.jsonl'

    print(f'Loading complete latent cache: {args.latent_dataset}', flush=True)
    cache = load_latent_cache(args.latent_dataset)
    embed_dim = int(cache.z.shape[1])
    goal_stride = args.action_block if args.goal_sampling == 'aligned_future' else 1
    train_episodes, val_episodes = split_episodes(
        len(cache.episode_offsets), args.train_fraction, args.split_seed
    )
    train_t, train_final = build_valid_transitions(
        cache.episode_offsets,
        cache.episode_lengths,
        train_episodes,
        min_future_steps=goal_stride,
    )
    val_t, val_final = build_valid_transitions(
        cache.episode_offsets,
        cache.episode_lengths,
        val_episodes,
        min_future_steps=goal_stride,
    )
    waypoint_steps = latent_path_waypoint_steps(args.subgoal_steps, args.action_block)
    goal_steps = ()
    if args.max_goal_steps is not None:
        goal_steps = tuple(
            range(goal_stride, args.max_goal_steps + 1, goal_stride)
        )
        train_t_table, train_history_table, train_counts = (
            build_distance_balanced_transition_tables(
                cache.episode_offsets,
                cache.episode_lengths,
                train_episodes,
                goal_steps,
                args.history_size,
            )
        )
        val_t_table, _, val_counts = build_distance_balanced_transition_tables(
            cache.episode_offsets,
            cache.episode_lengths,
            val_episodes,
            goal_steps,
            args.history_size,
        )
        fixed_val = sample_distance_balanced_future_pairs(
            val_t_table,
            val_counts,
            goal_steps,
            args.validation_pairs,
            args.subgoal_steps,
            seed=args.split_seed + 1,
        )
    else:
        fixed_val = sample_future_pairs(
            val_t,
            val_final,
            args.validation_pairs,
            args.subgoal_steps,
            seed=args.split_seed + 1,
            goal_stride=goal_stride,
        )
        train_history = build_history_indices(
            train_t, cache.episode_offsets, args.history_size
        )
    fixed_val_history = build_history_indices(
        fixed_val[0], cache.episode_offsets, args.history_size
    )
    validation_manifest_sha256 = None
    validation_manifest_metadata = None
    manifest_arrays = None
    if args.validation_manifest is not None:
        manifest_arrays, validation_manifest_metadata, validation_manifest_sha256 = (
            load_validation_manifest(args.validation_manifest, cache, val_episodes, args)
        )
        fixed_val = (
            manifest_arrays['current_indices'],
            manifest_arrays['goal_indices'],
            manifest_arrays['endpoint_target_indices'],
        )
        fixed_val_history = manifest_arrays['history_indices']

    if manifest_arrays is not None:
        target_indices = manifest_arrays['waypoint_target_indices']
    else:
        target_indices = np.stack(
            [np.minimum(fixed_val[0] + step, fixed_val[1]).astype(np.int32) for step in waypoint_steps],
            axis=1,
        )
    fixed_val = (fixed_val[0], fixed_val[1], target_indices)
    config_args = dict(vars(args))
    config = {
        **config_args,
        'latent_dataset': str(Path(args.latent_dataset).expanduser().resolve()),
        'save_dir': str(output_dir),
        'embed_dim': embed_dim,
        'num_rows': len(cache.z),
        'num_episodes': len(cache.episode_offsets),
        'num_train_episodes': len(train_episodes),
        'num_val_episodes': len(val_episodes),
        'num_train_transitions': len(train_t),
        'num_val_transitions': len(val_t),
        'lewm_checkpoint_sha256': cache.metadata.get('checkpoint_sha256'),
        'architecture': LATENT_PATH_FLOW_ARCHITECTURE,
        'goal_sampling': (
            f'uniform_distance_first_aligned_future_same_trajectory_'
            f'stride_{goal_stride}_max_{args.max_goal_steps}'
            if args.max_goal_steps is not None
            else 'hiql_uniform_future_same_trajectory'
            if args.goal_sampling == 'uniform_future'
            else f'uniform_aligned_future_same_trajectory_stride_{goal_stride}'
        ),
        'loss': 'conditional_path_flow_matching_mse',
        'conditioning': 'history_goal_time_adaln',
        'flow_solver': 'euler',
    }
    if args.validation_manifest is not None:
        config['validation_manifest'] = str(
            Path(args.validation_manifest).expanduser().resolve()
        )
        config['validation_manifest_sha256'] = validation_manifest_sha256
        config['validation_manifest_metadata'] = validation_manifest_metadata
    config_path = output_dir / 'config.json'
    if config_path.exists():
        existing_config = json.loads(config_path.read_text())
        if existing_config != json_safe(config):
            raise ValueError(f'Existing run config does not match requested config: {config_path}')
    else:
        write_json_atomic(config_path, config)

    print(
        f'Dataset rows={len(cache.z)} episodes={len(cache.episode_offsets)} '
        f'train_episodes={len(train_episodes)} val_episodes={len(val_episodes)} '
        f'train_transitions={len(train_t)} val_transitions={len(val_t)}',
        flush=True,
    )
    print(f'JAX backend={jax.default_backend()} devices={jax.devices()}', flush=True)

    z_device = jax.device_put(cache.z)
    if args.max_goal_steps is not None:
        train_t_device = jax.device_put(train_t_table)
        train_history_device = jax.device_put(train_history_table)
        train_final_device = jax.device_put(train_counts)
    else:
        train_t_device = jax.device_put(train_t)
        train_history_device = jax.device_put(train_history)
        train_final_device = jax.device_put(train_final)
    model = LatentPathFlow(
        embed_dim=embed_dim,
        num_waypoints=len(waypoint_steps),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        time_dim=args.time_dim,
        history_size=args.history_size,
    )
    init_rng, train_rng = jax.random.split(jax.random.PRNGKey(args.seed))
    empty_latents = jnp.zeros((1, embed_dim), dtype=jnp.float32)
    empty_current_latents = jnp.zeros((1, args.history_size, embed_dim), dtype=jnp.float32)
    variables = model.init(
        init_rng,
        jnp.zeros((1, len(waypoint_steps), embed_dim), dtype=jnp.float32),
        empty_current_latents,
        empty_latents,
        jnp.zeros((1,), dtype=jnp.float32),
    )
    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.train_steps,
        end_value=args.final_learning_rate,
    )
    decay_mask = jax.tree_util.tree_map(lambda value: value.ndim > 1, variables['params'])
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip),
        optax.adamw(
            learning_rate_schedule,
            weight_decay=args.weight_decay,
            mask=decay_mask,
        ),
    )
    state = FlowTrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        ema_params=variables['params'],
        tx=optimizer,
    )
    parameter_count = sum(value.size for value in jax.tree_util.tree_leaves(state.params))
    print(f'Model parameters={parameter_count:,}', flush=True)

    if args.resume:
        latest_step, latest_path = find_latest_checkpoint(output_dir)
        if latest_path is not None:
            state, train_rng = restore_checkpoint(state, latest_path)
            print(f'Resumed checkpoint step={latest_step}: {latest_path}', flush=True)

    train_step = make_train_step(
        model,
        learning_rate_schedule,
        args.batch_size,
        args.subgoal_steps,
        action_block=args.action_block,
        waypoint_steps=waypoint_steps,
        goal_stride=goal_stride,
        goal_steps=goal_steps,
        history_size=args.history_size,
        ema_decay=args.ema_decay,
    )
    predict_indices = make_predict_indices(
        model,
        flow_sampling_steps=args.flow_sampling_steps,
    )
    current_step = int(jax.device_get(state.step))
    if current_step > args.train_steps:
        raise ValueError(
            f'Checkpoint step {current_step} exceeds requested train_steps {args.train_steps}.'
        )

    started = time.monotonic()
    val_metrics = evaluate_validation(
        state.ema_params,
        z_device=z_device,
        z_host=cache.z,
        current_idxs=fixed_val[0],
        history_idxs=fixed_val_history,
        goal_idxs=fixed_val[1],
        target_idxs=fixed_val[2],
        predict_indices=predict_indices,
        batch_size=args.eval_batch_size,
        seed=args.split_seed + 2,
    )
    initial_row = {'type': 'validation', 'step': current_step, **val_metrics}
    append_jsonl(metrics_path, initial_row)
    print(f'Validation step={current_step}: {json.dumps(val_metrics, sort_keys=True)}', flush=True)

    latest_metrics = None
    while current_step < args.train_steps:
        state, train_rng, latest_metrics = train_step(
            state,
            train_rng,
            z_device,
            train_t_device,
            train_history_device,
            train_final_device,
        )
        current_step += 1
        if current_step % args.log_interval == 0:
            train_metrics = {
                key: float(value)
                for key, value in jax.device_get(latest_metrics).items()
            }
            row = {
                'type': 'train',
                'step': current_step,
                'elapsed_seconds': time.monotonic() - started,
                **train_metrics,
            }
            append_jsonl(metrics_path, row)
            print(f'Train step={current_step}: {json.dumps(train_metrics, sort_keys=True)}', flush=True)

        if current_step % args.eval_interval == 0 or current_step == args.train_steps:
            val_metrics = evaluate_validation(
                state.ema_params,
                z_device=z_device,
                z_host=cache.z,
                current_idxs=fixed_val[0],
                history_idxs=fixed_val_history,
                goal_idxs=fixed_val[1],
                target_idxs=fixed_val[2],
                predict_indices=predict_indices,
                batch_size=args.eval_batch_size,
                seed=args.split_seed + 2,
            )
            row = {
                'type': 'validation',
                'step': current_step,
                'elapsed_seconds': time.monotonic() - started,
                **val_metrics,
            }
            append_jsonl(metrics_path, row)
            print(f'Validation step={current_step}: {json.dumps(val_metrics, sort_keys=True)}', flush=True)

        if current_step % args.checkpoint_interval == 0 or current_step == args.train_steps:
            path = save_checkpoint(state, train_rng, output_dir)
            print(f'Saved checkpoint: {path}', flush=True)

    complete = {
        'status': 'complete',
        'step': current_step,
        'train_steps': args.train_steps,
        'elapsed_seconds': time.monotonic() - started,
        'final_validation': val_metrics,
        'final_checkpoint': str(checkpoint_path(output_dir, current_step)),
    }
    write_json_atomic(output_dir / 'complete.json', complete)
    print(f'Training complete: {json.dumps(complete, sort_keys=True)}', flush=True)


if __name__ == '__main__':
    main()
