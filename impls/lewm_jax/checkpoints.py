"""Loading helpers for frozen LeWM-JAX checkpoints."""

from __future__ import annotations

from pathlib import Path

import flax
import jax.numpy as jnp

from lewm_jax.model import LeWM

ARCHITECTURE = 'lewm_impala_small'


def load_frozen_lewm(checkpoint):
    """Restore a LeWM model and frozen variables from a checkpoint."""
    checkpoint = Path(checkpoint)
    payload = flax.serialization.msgpack_restore(checkpoint.read_bytes())
    config = payload['config']
    if config.get('architecture') != ARCHITECTURE:
        raise ValueError(f'Checkpoint architecture {config.get("architecture")!r} is not {ARCHITECTURE!r}.')
    try:
        dtype = {'bf16': jnp.bfloat16, 'float32': jnp.float32}[config.get('precision', 'bf16')]
    except KeyError as error:
        raise ValueError(f'Unsupported checkpoint precision: {config.get("precision")!r}.') from error
    model = LeWM(
        image_size=int(config['image_size']),
        embed_dim=int(config['embed_dim']),
        history_size=int(config['history_size']),
        projector_hidden_dim=int(config.get('projector_hidden_dim', 2048)),
        action_smoothed_dim=int(config.get('action_smoothed_dim', 10)),
        action_mlp_scale=int(config.get('action_mlp_scale', 4)),
        predictor_depth=int(config.get('predictor_depth', 6)),
        predictor_heads=int(config.get('predictor_heads', 16)),
        predictor_dim_head=int(config.get('predictor_dim_head', 64)),
        predictor_mlp_dim=int(config.get('predictor_mlp_dim', 2048)),
        predictor_dropout=float(config.get('predictor_dropout', 0.1)),
        predictor_emb_dropout=float(config.get('predictor_emb_dropout', 0.0)),
        dtype=dtype,
    )
    variables = {'params': payload['params'], 'batch_stats': payload['batch_stats']}
    metadata = {
        'path': str(checkpoint.resolve()),
        'epoch': int(payload['epoch']),
        'config': config,
    }
    return model, variables, metadata
