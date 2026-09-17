"""Train the final-goal-conditioned Action-Prior-Chunk used by LeWM++."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from agents.action_prior_chunk import (
    REPRESENTATION_MODES,
    ActionPriorChunkAgent,
    get_config,
    representation_sharing,
)
from lewm_jax import load_frozen_lewm
from utils.datasets import Dataset, GCChunkDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import save_agent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env_name')
    parser.add_argument('--dataset_path')
    parser.add_argument('--lewm_checkpoint', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--train_steps', type=int, default=100_000)
    parser.add_argument('--save_interval', type=int, default=100_000)
    parser.add_argument('--log_interval', type=int, default=5_000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--expectile', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--chunk_size', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--p_aug', type=float, default=0.0)
    parser.add_argument('--validation_fraction', type=float, default=0.05)
    parser.add_argument('--representation_mode', choices=tuple(REPRESENTATION_MODES), default='all')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.env_name is None and args.dataset_path is None:
        raise ValueError('Provide --env_name or --dataset_path.')
    if not 0.0 <= args.p_aug <= 1.0:
        raise ValueError('--p_aug must be in [0, 1].')
    for name in ('train_steps', 'save_interval', 'log_interval', 'batch_size', 'chunk_size'):
        if getattr(args, name) < 1:
            raise ValueError(f'--{name} must be positive.')
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError('--validation_fraction must be in [0, 1).')

    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, lewm_variables, lewm_metadata = load_frozen_lewm(args.lewm_checkpoint)
    lewm_sha256 = hashlib.sha256(Path(args.lewm_checkpoint).expanduser().read_bytes()).hexdigest()

    config = get_config()
    config.lr = args.lr
    config.batch_size = args.batch_size
    config.discount = args.discount
    config.expectile = args.expectile
    config.tau = args.tau
    config.chunk_size = args.chunk_size
    config.actor_loss = 'awr'
    config.alpha = args.alpha
    # Augmentation is applied once in prepare_batch, before either pixel or
    # frozen-LeWM encoding, so all representation modes receive the same view.
    config.p_aug = 0.0
    config.frame_stack = None
    sharing = representation_sharing(args.representation_mode)
    for module, shared in sharing.items():
        config[f'share_{module}_encoder'] = shared
    config.representation_mode = args.representation_mode
    config.latent_dim = int(lewm_metadata['config']['embed_dim'])

    _, train_base, val_base = make_env_and_datasets(
        args.env_name or 'dataset-only',
        dataset_path=args.dataset_path,
        validation_fraction=args.validation_fraction,
    )
    if not getattr(train_base, 'lazy', False):
        train_base = Dataset.create(**train_base)
    if val_base is not None and not getattr(val_base, 'lazy', False):
        val_base = Dataset.create(**val_base)
    train_dataset = GCChunkDataset(train_base, config, preprocess_frame_stack=False)
    val_dataset = None if val_base is None else GCChunkDataset(val_base, config, preprocess_frame_stack=False)

    np.random.seed(args.seed)
    example = train_dataset.sample(1, evaluation=True)
    agent = ActionPriorChunkAgent.create(
        args.seed,
        jnp.asarray(example['observations']),
        jnp.zeros((1, config.latent_dim), dtype=jnp.float32),
        jnp.asarray(example['actions'], dtype=jnp.float32),
        config,
    )

    @jax.jit
    def encode_pixels(pixels):
        return model.apply(
            lewm_variables,
            pixels,
            train=False,
            method=model.encode_pixels,
        ).astype(jnp.float32)

    pixel_keys = (
        'observations',
        'next_observations',
        'value_goals',
        'actor_goals',
    )

    def prepare_batch(sample, training):
        batch = {key: np.array(sample[key], copy=True) for key in pixel_keys}
        if training and args.p_aug > 0 and np.random.rand() < args.p_aug:
            train_dataset.augment(batch, list(pixel_keys))
        batch = {key: jnp.asarray(value) for key, value in batch.items()}
        batch_size = batch['observations'].shape[0]
        latents = encode_pixels(jnp.concatenate([batch[key] for key in pixel_keys], axis=0))
        latent_parts = jnp.split(
            latents,
            tuple(batch_size * i for i in range(1, len(pixel_keys))),
            axis=0,
        )
        batch.update({f'lewm_{key}': value for key, value in zip(pixel_keys, latent_parts)})
        for key in ('actions', 'rewards', 'masks'):
            batch[key] = jnp.asarray(sample[key], dtype=jnp.float32)
        return batch

    metadata = {
        'entrypoint': 'train_action_prior_chunk.py',
        'env_name': args.env_name,
        'dataset_path': args.dataset_path,
        'lewm_checkpoint': lewm_metadata['path'],
        'lewm_checkpoint_sha256': lewm_sha256,
        'train_steps': args.train_steps,
        'seed': args.seed,
        'agent': dict(config),
        'representation': {
            'mode': args.representation_mode,
            'module': 'LeWM.encode_pixels',
            'output': 'post_projector',
            'latent_dim': int(config.latent_dim),
            'frozen': True,
            **{module: 'lewm' if shared else 'pixel' for module, shared in sharing.items()},
            'downstream_heads_shared': False,
            'lewm_checkpoint': lewm_metadata['path'],
        },
        'augmentation': {
            'probability': args.p_aug,
            'applied_before_all_encoders': True,
        },
    }
    (output_dir / 'flags.json').write_text(json.dumps(metadata, indent=2) + '\n')

    fields = [
        'step',
        'loss',
        'value/value_loss',
        'critic/critic_loss',
        'actor/actor_loss',
        'actor/adv',
        'actor/mse',
        'actor/std',
        'validation/loss',
        'steps_per_second',
        'total_seconds',
    ]
    started = time.time()
    interval_started = started
    with (output_dir / 'train.csv').open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for step in range(1, args.train_steps + 1):
            batch = prepare_batch(train_dataset.sample(args.batch_size), training=True)
            agent, info = agent.update(batch)

            if step == 1 or step % args.log_interval == 0:
                elapsed = time.time() - interval_started
                val_loss = None
                if val_dataset is not None:
                    val_batch = prepare_batch(
                        val_dataset.sample(args.batch_size, evaluation=True),
                        training=False,
                    )
                    val_loss, _ = agent.total_loss(val_batch, grad_params=None)
                row = {key: float(value) for key, value in jax.device_get(info).items()}
                row.update(
                    {
                        'step': step,
                        'loss': sum(
                            row[key]
                            for key in (
                                'value/value_loss',
                                'critic/critic_loss',
                                'actor/actor_loss',
                            )
                        ),
                        'validation/loss': ('' if val_loss is None else float(jax.device_get(val_loss))),
                        'steps_per_second': ((1 if step == 1 else args.log_interval) / elapsed),
                        'total_seconds': time.time() - started,
                    }
                )
                writer.writerow(row)
                csv_file.flush()
                print(json.dumps(row), flush=True)
                interval_started = time.time()

            if step % args.save_interval == 0 or step == args.train_steps:
                save_agent(agent, output_dir, step)


if __name__ == '__main__':
    main()
