"""Inference runtime for the public LeWM++ LatentPathFlow generator."""

from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from latent_path_flow_lewm_control import (
    load_checkpoint,
    sample_path,
    waypoint_steps,
)


class SubgoalGenerator:
    """Load one LatentPathFlow checkpoint and predict one path per replan."""

    def __init__(
        self,
        checkpoint,
        encode_pixels,
        *,
        seed,
        action_block,
        lewm_checkpoint,
        flow_sampling_steps=16,
    ):
        if int(flow_sampling_steps) <= 0:
            raise ValueError('Flow sampling steps must be positive.')

        model, params, config, checkpoint_step = load_checkpoint(checkpoint)
        digest = hashlib.sha256()
        with Path(lewm_checkpoint).expanduser().open('rb') as file:
            for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        expected_sha = config.get('lewm_checkpoint_sha256')
        if expected_sha is not None and digest.hexdigest() != expected_sha:
            raise ValueError('Subgoal generator and controller must use the same frozen LeWM.')

        trained_action_block = int(config.get('action_block', action_block))
        if trained_action_block != int(action_block):
            raise ValueError(
                f'Generator and controller action blocks must match: {trained_action_block} != {action_block}.'
            )
        steps = waypoint_steps(config['subgoal_steps'], trained_action_block)

        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.checkpoint_step = int(checkpoint_step)
        self.config = config
        self.generator_type = 'latent_path_flow'
        self.encode_pixels = encode_pixels
        self.seed = int(seed)
        self.num_samples = 1
        self.embed_dim = int(config['embed_dim'])
        self.waypoint_step = int(config['subgoal_steps'])
        self.waypoint_index = steps.index(self.waypoint_step)
        self.path_length = len(steps)
        self.history_size = int(config.get('history_size', 1))
        self.sample_selection = 'single_sample'
        self.flow_sampling_steps = int(flow_sampling_steps)
        self._predict = jax.jit(
            lambda history, goal, rng: sample_path(
                model,
                params,
                history,
                goal,
                rng,
                num_steps=self.flow_sampling_steps,
            )
        )
        self.histories = []
        self.generation_counts = np.zeros(0, dtype=np.int64)

    def reset(self, num_envs):
        self.histories = [deque(maxlen=self.history_size) for _ in range(num_envs)]
        self.generation_counts = np.zeros(num_envs, dtype=np.int64)

    def observe(self, env_index, pixels):
        self.histories[env_index].append(np.asarray(pixels))

    def predict_path(self, env_index, goal_pixels):
        history = list(self.histories[env_index])
        if not history:
            raise ValueError('A current observation is required before path prediction.')
        history = [history[0]] * (self.history_size - len(history)) + history
        history_embeddings = np.asarray(self.encode_pixels(jnp.asarray(np.stack(history))))[None]
        goal_embedding = np.asarray(self.encode_pixels(jnp.asarray(np.asarray(goal_pixels)[None])))
        generation = int(self.generation_counts[env_index])
        rng = jax.random.fold_in(jax.random.PRNGKey(self.seed), int(env_index))
        rng = jax.random.fold_in(rng, generation)
        prediction = np.asarray(self._predict(jnp.asarray(history_embeddings), jnp.asarray(goal_embedding), rng))[
            0
        ].astype(np.float32)
        expected_shape = (self.path_length, self.embed_dim)
        if prediction.shape != expected_shape or not np.isfinite(prediction).all():
            raise FloatingPointError(f'Invalid latent path {prediction.shape}; expected {expected_shape}.')
        self.generation_counts[env_index] += 1
        return prediction
