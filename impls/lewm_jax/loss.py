"""LeWM prediction and SIGReg objectives."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sigreg_loss(embeddings, key, *, knots=17, num_proj=1024):
    """Single-device Sketch Isotropic Gaussian Regularizer."""
    # Match PyTorch bf16 autocast: the projection matmul runs in bf16, the
    # multiply by fp32 quadrature points promotes the trigonometric section to
    # fp32, and the final matrix multiply returns to the embedding dtype.
    projection = embeddings.transpose(1, 0, 2)
    directions = jax.random.normal(key, (projection.shape[-1], num_proj), dtype=jnp.float32)
    directions = directions / jnp.linalg.norm(directions, axis=0, keepdims=True)
    t = jnp.linspace(0.0, 3.0, knots, dtype=jnp.float32)
    dt = 3.0 / (knots - 1)
    weights = jnp.full((knots,), 2.0 * dt, dtype=jnp.float32)
    weights = weights.at[jnp.asarray([0, knots - 1])].set(dt)
    phi = jnp.exp(-(t**2) / 2.0)
    weights = weights * phi
    projected = jnp.einsum('tbd,dp->tbp', projection, directions.astype(projection.dtype))
    x_t = projected[..., None] * t
    error = (jnp.cos(x_t).mean(axis=1) - phi) ** 2
    error = error + jnp.sin(x_t).mean(axis=1) ** 2
    statistic = (
        jnp.einsum(
            'tpk,k->tp',
            error.astype(projection.dtype),
            weights.astype(projection.dtype),
        )
        * projection.shape[1]
    )
    return statistic.mean()


def lewm_loss(
    model,
    variables,
    batch,
    *,
    train,
    dropout_key,
    sigreg_key,
    sigreg_weight=0.09,
    sigreg_knots=17,
    sigreg_num_proj=1024,
):
    """Return scalar loss, metrics, and updated BatchNorm statistics."""
    if train:
        (embeddings, predictions), updates = model.apply(
            variables,
            batch['pixels'],
            jnp.nan_to_num(batch['action']),
            train=True,
            rngs={'dropout': dropout_key},
            mutable=['batch_stats'],
        )
    else:
        embeddings, predictions = model.apply(
            variables,
            batch['pixels'],
            jnp.nan_to_num(batch['action']),
            train=False,
            rngs={'dropout': dropout_key},
            mutable=False,
        )
        updates = {'batch_stats': variables['batch_stats']}

    targets = embeddings[:, 1:]
    prediction_loss = jnp.mean((predictions - targets) ** 2)
    regularizer = sigreg_loss(
        embeddings,
        sigreg_key,
        knots=sigreg_knots,
        num_proj=sigreg_num_proj,
    )
    loss = prediction_loss + sigreg_weight * regularizer
    metrics = {
        'loss': loss,
        'pred_loss': prediction_loss,
        'sigreg_loss': regularizer,
    }
    return loss, (metrics, updates['batch_stats'])
