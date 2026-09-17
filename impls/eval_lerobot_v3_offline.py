"""Evaluate trained LeWM++ components on held-out LeRobot v3 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lance-path', required=True)
    parser.add_argument('--lewm-checkpoint', required=True)
    parser.add_argument('--action-prior-dir', required=True)
    parser.add_argument('--action-prior-step', type=int, required=True)
    parser.add_argument('--latent-dataset', required=True)
    parser.add_argument('--latent-path-flow-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-fraction', type=float, default=0.96)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--eval-seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--action-prior-samples', type=int, default=10_000)
    parser.add_argument('--flow-validation-pairs', type=int, default=10_000)
    parser.add_argument('--flow-batch-size', type=int, default=1024)
    return parser.parse_args()


def evaluate_lewm(args):
    from lewm_jax import load_frozen_lewm
    from utils.lewm_sequence_dataset import LeWMSequenceDataset

    model, variables, metadata = load_frozen_lewm(args.lewm_checkpoint)
    config = metadata['config']
    dataset = LeWMSequenceDataset(
        args.lance_path,
        num_steps=int(config['history_size']) + int(config.get('num_preds', 1)),
        frameskip=int(config['frameskip']),
        train_fraction=args.train_fraction,
        seed=int(config['seed']),
        decode_workers=6,
        normalize_pixels=False,
        episode_split=True,
        split_seed=args.split_seed,
    )

    @jax.jit
    def predict(pixels, actions):
        return model.apply(variables, pixels, actions, train=False)

    squared_error = 0.0
    cosine_sum = 0.0
    prediction_count = 0
    try:
        for start in range(0, len(dataset.val_indices), args.batch_size):
            indices = dataset.val_indices[start : start + args.batch_size]
            batch = dataset.get_batch(indices)
            embeddings, predictions = predict(jnp.asarray(batch['pixels']), jnp.asarray(batch['action']))
            predictions = np.asarray(jax.device_get(predictions), dtype=np.float32)
            targets = np.asarray(jax.device_get(embeddings[:, 1:]), dtype=np.float32)
            squared_error += float(np.square(predictions - targets).sum())
            flat_predictions = predictions.reshape(-1, predictions.shape[-1])
            flat_targets = targets.reshape(-1, targets.shape[-1])
            cosine_sum += float(
                np.sum(
                    np.sum(flat_predictions * flat_targets, axis=-1)
                    / (
                        np.linalg.norm(flat_predictions, axis=-1)
                        * np.linalg.norm(flat_targets, axis=-1)
                        + 1e-8
                    )
                )
            )
            prediction_count += len(flat_predictions)
    finally:
        dataset.close()
    return {
        'checkpoint': metadata['path'],
        'validation_episodes': int(len(dataset.val_episode_indices)),
        'validation_clips': int(len(dataset.val_indices)),
        'latent_prediction_mse': squared_error / (prediction_count * int(config['embed_dim'])),
        'latent_prediction_cosine_similarity': cosine_sum / prediction_count,
    }


def evaluate_action_prior(args):
    from action_prior_runtime_lewm_control import load_action_prior
    from utils.datasets import GCChunkDataset
    from utils.lewm_dataset import LeWMLanceDataset

    prior = load_action_prior(
        args.lance_path,
        args.action_prior_dir,
        args.action_prior_step,
        lewm_checkpoint=args.lewm_checkpoint,
        expected_representation_mode='all',
    )
    base = LeWMLanceDataset(
        args.lance_path,
        split='val',
        validation_fraction=1.0 - args.train_fraction,
        episode_split_seed=args.split_seed,
    )
    dataset = GCChunkDataset(base, prior.agent.config, preprocess_frame_stack=False)
    np.random.seed(args.eval_seed)
    rng = np.random.default_rng(args.eval_seed)
    squared_error = 0.0
    absolute_error = 0.0
    value_count = 0
    completed = 0
    while completed < args.action_prior_samples:
        count = min(args.batch_size, args.action_prior_samples - completed)
        indices = dataset.chunk_valid_idxs[rng.integers(len(dataset.chunk_valid_idxs), size=count)]
        batch = dataset.sample(count, idxs=indices, evaluation=True)
        key = jax.random.fold_in(jax.random.PRNGKey(args.eval_seed), completed)
        predictions = np.asarray(
            jax.device_get(
                prior.sample_actions(
                    batch['observations'],
                    batch['actor_goals'],
                    seed=key,
                    temperature=0.0,
                )
            ),
            dtype=np.float32,
        )
        targets = np.asarray(batch['actions'], dtype=np.float32)
        difference = predictions - targets
        squared_error += float(np.square(difference).sum())
        absolute_error += float(np.abs(difference).sum())
        value_count += difference.size
        completed += count
    return {
        'checkpoint_dir': str(Path(args.action_prior_dir).expanduser().resolve()),
        'checkpoint_step': args.action_prior_step,
        'validation_episodes': int(len(base.selected_episode_indices)),
        'samples': args.action_prior_samples,
        'normalized_action_mse': squared_error / value_count,
        'normalized_action_mae': absolute_error / value_count,
    }


def evaluate_latent_path_flow(args):
    from latent_path_flow_lewm_control import load_checkpoint, waypoint_steps
    from train_latent_path_flow_lewm_control import evaluate_validation, make_predict_indices
    from utils.latent_path_flow_dataset_lewm_control import (
        build_history_indices,
        build_valid_transitions,
        load_latent_cache,
        sample_future_pairs,
        split_episodes,
    )

    cache = load_latent_cache(args.latent_dataset)
    model, params, config, step = load_checkpoint(args.latent_path_flow_checkpoint)
    if float(config['train_fraction']) != args.train_fraction or int(config['split_seed']) != args.split_seed:
        raise ValueError('LatentPathFlow checkpoint does not use the requested held-out episode split.')
    _, val_episodes = split_episodes(len(cache.episode_offsets), args.train_fraction, args.split_seed)
    val_t, val_final = build_valid_transitions(
        cache.episode_offsets,
        cache.episode_lengths,
        val_episodes,
        min_future_steps=1,
    )
    current, goal, _ = sample_future_pairs(
        val_t,
        val_final,
        args.flow_validation_pairs,
        int(config['subgoal_steps']),
        seed=args.eval_seed,
    )
    history = build_history_indices(current, cache.episode_offsets, int(config['history_size']))
    offsets = waypoint_steps(config['subgoal_steps'], config['action_block'])
    targets = np.stack([np.minimum(current + offset, goal) for offset in offsets], axis=1)
    metrics = evaluate_validation(
        params,
        z_device=jax.device_put(cache.z),
        z_host=cache.z,
        current_idxs=current,
        history_idxs=history,
        goal_idxs=goal,
        target_idxs=targets,
        predict_indices=make_predict_indices(model, flow_sampling_steps=int(config['flow_sampling_steps'])),
        batch_size=args.flow_batch_size,
        seed=args.eval_seed + 1,
    )
    return {
        'checkpoint': str(Path(args.latent_path_flow_checkpoint).expanduser().resolve()),
        'checkpoint_step': step,
        'validation_episodes': int(len(val_episodes)),
        'validation_pairs': args.flow_validation_pairs,
        **metrics,
    }


def main():
    args = parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError('--train-fraction must be in (0, 1).')
    for name in ('batch_size', 'action_prior_samples', 'flow_validation_pairs', 'flow_batch_size'):
        if getattr(args, name) <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive.')
    results = {
        'protocol': {
            'dataset': 'yaoxianze/push_multi_red_cube',
            'train_fraction': args.train_fraction,
            'split_seed': args.split_seed,
            'eval_seed': args.eval_seed,
        },
        'lewm': evaluate_lewm(args),
        'action_prior': evaluate_action_prior(args),
        'latent_path_flow': evaluate_latent_path_flow(args),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + '\n')
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
