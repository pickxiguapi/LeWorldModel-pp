"""Checkpoint configuration and LeWM encoding for Visual OGBench action priors."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp


def load_agent_config(checkpoint_dir):
    """Restore the saved configuration for an Action Chunk Prior checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    saved = json.loads((checkpoint_dir / 'flags.json').read_text())
    saved_agent = saved.get('agent', {})
    name = saved_agent.get('agent_name')
    if name == 'gciql_chunk':
        from agents.action_prior_chunk_ogbench import get_config
    elif name == 'gciql_chunk_lewm':
        from agents.action_prior_chunk_lewm_ogbench import get_config
    else:
        raise ValueError(f'Expected an Action Chunk Prior checkpoint, got {name!r}.')

    config = get_config()
    for key, value in saved_agent.items():
        if key in config:
            config[key] = value
    return name, config, saved


class LeWMEncodedAgent:
    """Route actor calls through their configured pixel or LeWM inputs."""

    def __init__(
        self,
        agent,
        encode_pixels,
        share_pi_encoder,
        lewm_checkpoint=None,
    ):
        self.agent = agent
        self.encode_pixels = encode_pixels
        self.share_pi_encoder = bool(share_pi_encoder)
        self.lewm_checkpoint = lewm_checkpoint
        self.action_horizon = int(agent.action_horizon)

    def sample_actions(self, observations, goals, seed, temperature):
        if self.share_pi_encoder:
            observations = self.encode_pixels(jnp.asarray(observations))
            goals = self.encode_pixels(jnp.asarray(goals))
        return self.agent.sample_actions(
            observations=observations,
            goals=goals,
            seed=seed,
            temperature=temperature,
        )
