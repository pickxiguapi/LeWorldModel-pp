"""Load the final-goal-conditioned Action-Prior-Chunk used by LeWM++."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def load_action_prior(
    lance_path,
    checkpoint_dir,
    checkpoint_step,
    lewm_checkpoint=None,
    expected_representation_mode=None,
):
    """Restore an action prior and the frozen LeWM encoder used by its shared branches."""
    from agents.action_prior_chunk import (
        ActionPriorChunkAgent,
        get_config,
        resolve_representation_mode,
        validate_representation_sharing,
    )
    from lewm_jax import load_frozen_lewm
    from utils.datasets import GCChunkDataset
    from utils.flax_utils import restore_agent
    from utils.lewm_dataset import LeWMLanceDataset

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    flags_path = checkpoint_dir / 'flags.json'
    if not flags_path.is_file():
        raise FileNotFoundError(f'Action-prior metadata not found: {flags_path}')
    saved = json.loads(flags_path.read_text())
    saved_agent = saved.get('agent', {})
    representation = saved.get('representation', {})
    representation_mode = resolve_representation_mode(representation, saved_agent)
    if expected_representation_mode is not None and representation_mode != expected_representation_mode:
        raise ValueError(
            f'Action-prior checkpoint uses representation mode {representation_mode!r}, '
            f'but evaluation requested {expected_representation_mode!r}.'
        )

    config = get_config()
    for key, value in saved_agent.items():
        if key in config and key != 'agent_name':
            config[key] = value
    config.representation_mode = representation_mode
    expected_sharing = validate_representation_sharing(representation_mode, config, label='metadata')
    for module, shared in expected_sharing.items():
        recorded_source = representation.get(module)
        expected_source = 'lewm' if shared else 'pixel'
        if recorded_source not in (None, expected_source):
            raise ValueError(
                f'Action-prior representation metadata is inconsistent: {module}={recorded_source!r}, '
                f'expected {expected_source!r} for mode={representation_mode!r}.'
            )

    base = LeWMLanceDataset(lance_path, split='train', validation_fraction=0.05)
    dataset = GCChunkDataset(base, config, preprocess_frame_stack=False)
    example = dataset.sample(1, evaluation=True)
    lewm_checkpoint = lewm_checkpoint or saved.get('lewm_checkpoint') or representation.get('lewm_checkpoint')
    if lewm_checkpoint is None:
        raise ValueError('A frozen LeWM checkpoint is required to restore the action prior.')
    expected_sha = saved.get('lewm_checkpoint_sha256')
    if expected_sha is not None:
        actual_sha = hashlib.sha256(Path(lewm_checkpoint).expanduser().read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError('Action prior and evaluator must use the same frozen LeWM checkpoint.')
    model, variables, metadata = load_frozen_lewm(lewm_checkpoint)
    agent = ActionPriorChunkAgent.create(
        0,
        jnp.asarray(example['observations']),
        jnp.zeros((1, int(metadata['config']['embed_dim'])), dtype=jnp.float32),
        jnp.asarray(example['actions'], dtype=jnp.float32),
        config,
    )
    agent = restore_agent(agent, checkpoint_dir, checkpoint_step)
    encode_pixels = jax.jit(
        lambda pixels: model.apply(
            variables,
            pixels,
            train=False,
            method=model.encode_pixels,
        ).astype(jnp.float32)
    )
    return FinalGoalActionPrior(agent, encode_pixels, metadata['path'], representation_mode)


class FinalGoalActionPrior:
    """Adapter exposing deterministic normalized action chunks to CEM."""

    def __init__(self, agent, encode_pixels, lewm_checkpoint, representation_mode='all'):
        from agents.action_prior_chunk import representation_sharing

        self.agent = agent
        self.encode_pixels = encode_pixels
        self.lewm_checkpoint = lewm_checkpoint
        self.representation_mode = representation_mode
        self.share_pi_encoder = representation_sharing(representation_mode)['pi']
        self.action_horizon = int(agent.action_horizon)

    def sample_actions(self, observations, goals, seed, temperature=0.0):
        observations = jnp.asarray(observations)
        goals = jnp.asarray(goals)
        if self.share_pi_encoder:
            observations = self.encode_pixels(observations)
            goals = self.encode_pixels(goals)
        return self.agent.sample_actions(
            observations=observations,
            goals=goals,
            seed=seed,
            temperature=temperature,
        )


class FinalGoalPolicy:
    """Execute Action-Prior-Chunk directly, always conditioned on the final goal."""

    def __init__(self, action_prior, scaler, seed):
        self.action_prior = action_prior
        self.scaler = scaler
        self.seed = int(seed)
        self.action_horizon = int(action_prior.action_horizon)

    def reset(self, action_space, num_envs):
        self.action_dim = int(np.prod(action_space.shape))
        self.buffers = [deque() for _ in range(num_envs)]
        self.chunk_counts = np.zeros(num_envs, dtype=np.int64)

    def get_actions(self, pixels, goals, alive):
        for env_index in np.flatnonzero(alive):
            if self.buffers[env_index]:
                continue
            key = jax.random.fold_in(jax.random.PRNGKey(self.seed), int(env_index))
            key = jax.random.fold_in(key, int(self.chunk_counts[env_index]))
            normalized = np.asarray(
                self.action_prior.sample_actions(
                    observations=np.asarray(pixels[env_index, -1:]),
                    goals=np.asarray(goals[env_index, -1:]),
                    seed=key,
                    temperature=0.0,
                )
            )[0]
            expected = self.action_horizon * self.action_dim
            if normalized.shape != (expected,):
                raise ValueError(f'Action prior returned {normalized.shape}; expected ({expected},).')
            actions = self.scaler.inverse_transform(normalized.reshape(self.action_horizon, self.action_dim))
            self.buffers[env_index].extend(actions)
            self.chunk_counts[env_index] += 1

        result = np.full((len(alive), self.action_dim), np.nan, dtype=np.float32)
        for env_index in np.flatnonzero(alive):
            result[env_index] = self.buffers[env_index].popleft()
        return result
