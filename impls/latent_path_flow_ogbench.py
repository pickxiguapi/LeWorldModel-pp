"""Visual OGBench LatentPathFlow model and checkpoint loader."""

from __future__ import annotations

import json
from pathlib import Path

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

LATENT_PATH_FLOW_ARCHITECTURE = 'latent_path_flow_transformer_encoder'


def latent_path_waypoint_steps(subgoal_steps, action_block):
    """Derive path prediction offsets from the control chunk granularity."""
    subgoal_steps = int(subgoal_steps)
    action_block = int(action_block)
    if subgoal_steps <= 0 or action_block <= 0:
        raise ValueError('Subgoal steps and action block must be positive.')
    if subgoal_steps % action_block:
        raise ValueError('Subgoal steps must be divisible by the action block.')
    return tuple(range(action_block, subgoal_steps + 1, action_block))


def sinusoidal_time_embedding(times, dim):
    """Embed unit-interval flow times with fixed sinusoidal features."""
    if dim <= 0 or dim % 2:
        raise ValueError('Flow-time embedding dimension must be positive and even.')
    times = jnp.asarray(times, dtype=jnp.float32).reshape(-1, 1)
    frequencies = jnp.exp(
        -jnp.log(10_000.0)
        * jnp.arange(dim // 2, dtype=jnp.float32)
        / max(dim // 2 - 1, 1)
    )
    angles = times * frequencies[None]
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)


class AdaLNTransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer block modulated by one global condition vector."""

    model_dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, tokens, condition):
        modulation = nn.Dense(
            4 * self.model_dim,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros_init(),
            name='condition_modulation',
        )(nn.silu(condition))
        attention_scale, attention_shift, mlp_scale, mlp_shift = jnp.split(modulation, 4, axis=-1)

        residual = tokens
        tokens = nn.LayerNorm(use_scale=False, use_bias=False, name='attention_norm')(tokens)
        tokens = tokens * (1.0 + attention_scale[:, None]) + attention_shift[:, None]
        tokens = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.model_dim,
            out_features=self.model_dim,
            dropout_rate=0.0,
            deterministic=True,
            name='self_attention',
        )(tokens)
        tokens = residual + tokens

        residual = tokens
        tokens = nn.LayerNorm(use_scale=False, use_bias=False, name='mlp_norm')(tokens)
        tokens = tokens * (1.0 + mlp_scale[:, None]) + mlp_shift[:, None]
        tokens = nn.Dense(self.mlp_dim, name='mlp_in')(tokens)
        tokens = nn.gelu(tokens, approximate=False)
        tokens = nn.Dense(self.model_dim, name='mlp_out')(tokens)
        return residual + tokens


class LatentPathFlow(nn.Module):
    """History-conditioned vector field over a short latent waypoint path."""

    embed_dim: int
    num_waypoints: int = 2
    hidden_dim: int = 512
    depth: int = 4
    num_heads: int = 8
    ff_dim: int = 2048
    time_dim: int = 64
    history_size: int = 3

    @nn.compact
    def __call__(self, noisy_path, current_latents, goal_latents, flow_times):
        if noisy_path.ndim != 3 or noisy_path.shape[1] != self.num_waypoints:
            raise ValueError(f'Expected noisy path [B, {self.num_waypoints}, {self.embed_dim}].')
        if self.history_size <= 1:
            raise ValueError('Release LatentPathFlow checkpoints require history_size > 1.')
        if current_latents.ndim != 3 or current_latents.shape[1] != self.history_size:
            raise ValueError(f'Expected history [B, {self.history_size}, {self.embed_dim}].')

        path_tokens = nn.Dense(self.hidden_dim, name='path_projection')(noisy_path)
        path_positions = self.param(
            'path_position_embeddings',
            nn.initializers.normal(stddev=0.02),
            (self.num_waypoints, self.hidden_dim),
        )
        path_tokens = path_tokens + path_positions[None]
        history_condition = nn.Dense(self.hidden_dim, name='history_condition_projection')(
            current_latents.reshape(current_latents.shape[0], -1)
        )
        goal_condition = nn.Dense(self.hidden_dim, name='goal_condition_projection')(goal_latents)
        time_condition = nn.Dense(self.hidden_dim, name='time_projection_in')(
            sinusoidal_time_embedding(flow_times, self.time_dim)
        )
        time_condition = nn.Dense(self.hidden_dim, name='time_projection_out')(nn.silu(time_condition))
        condition = nn.Dense(self.hidden_dim, name='condition_fusion_in')(
            jnp.concatenate((history_condition, goal_condition, time_condition), axis=-1)
        )
        condition = nn.Dense(self.hidden_dim, name='condition_fusion_out')(nn.silu(condition))

        for layer_index in range(self.depth):
            path_tokens = AdaLNTransformerEncoderBlock(
                model_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.ff_dim,
                name=f'encoder_block_{layer_index}',
            )(path_tokens, condition)
        path_tokens = nn.LayerNorm(name='output_norm')(path_tokens)
        return nn.Dense(self.embed_dim, name='velocity_head')(path_tokens)


def sample_conditional_path_flow(
    model,
    params,
    current_latents,
    goal_latents,
    rng,
    *,
    num_steps=16,
):
    """Integrate one latent path from Gaussian noise with Euler steps."""
    if num_steps <= 0:
        raise ValueError('Flow sampling steps must be positive.')
    current_latents = jnp.asarray(current_latents, dtype=jnp.float32)
    goal_latents = jnp.asarray(goal_latents, dtype=jnp.float32)
    sample_shape = (
        current_latents.shape[0],
        int(model.num_waypoints),
        current_latents.shape[-1],
    )
    samples = jax.random.normal(rng, sample_shape, dtype=jnp.float32)
    step_size = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)

    def integrate_step(index, value):
        flow_time = jnp.full((value.shape[0],), index.astype(jnp.float32) * step_size)
        velocity = model.apply({'params': params}, value, current_latents, goal_latents, flow_time)
        return value + step_size * velocity

    return jax.lax.fori_loop(0, num_steps, integrate_step, samples)


def load_latent_subgoal_checkpoint(path):
    """Load one release LatentPathFlow checkpoint and its adjacent config."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Latent subgoal checkpoint not found: {path}')
    config_path = path.parent / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Latent subgoal config must be adjacent to its checkpoint: {config_path}')
    config = json.loads(config_path.read_text())
    if config.get('architecture') != LATENT_PATH_FLOW_ARCHITECTURE:
        raise ValueError(f'Unsupported latent subgoal architecture: {config.get("architecture")!r}.')
    if config.get('loss') != 'conditional_path_flow_matching_mse':
        raise ValueError(f'Unsupported latent path flow loss: {config.get("loss")!r}.')
    if config.get('conditioning') != 'history_goal_time_adaln':
        raise ValueError('Release LatentPathFlow requires history_goal_time_adaln conditioning.')
    if config.get('flow_solver') != 'euler':
        raise ValueError('Release LatentPathFlow checkpoints must use Euler integration.')
    history_size = int(config.get('history_size', 0))
    if history_size <= 1:
        raise ValueError('Release LatentPathFlow checkpoints require history_size > 1.')
    waypoint_steps = latent_path_waypoint_steps(config['subgoal_steps'], config['action_block'])
    model = LatentPathFlow(
        embed_dim=int(config['embed_dim']),
        num_waypoints=len(waypoint_steps),
        hidden_dim=int(config['hidden_dim']),
        depth=int(config['depth']),
        num_heads=int(config['num_heads']),
        ff_dim=int(config['ff_dim']),
        time_dim=int(config['time_dim']),
        history_size=history_size,
    )

    payload = flax.serialization.msgpack_restore(path.read_bytes())
    if set(payload) != {'rng', 'step', 'train_state'}:
        raise ValueError(f'Unexpected latent subgoal checkpoint keys: {set(payload)}.')
    state = payload['train_state']
    if 'params' not in state or int(state['step']) != int(payload['step']):
        raise ValueError(f'Invalid latent subgoal train state in {path}.')
    if int(payload['step']) <= 0:
        raise ValueError(f'Latent subgoal checkpoint step must be positive: {path}.')
    return model, state.get('ema_params', state['params']), config, int(payload['step'])
