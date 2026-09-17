"""LeWM with OGBench's native IMPALA-small visual encoder."""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from utils.encoders import encoder_modules

from lewm_jax.modules import ActionEmbedder, ARPredictor, ProjectionMLP


class LeWM(nn.Module):
    """End-to-end IMPALA-small encoder plus LeWM predictor."""

    image_size: int = 224
    embed_dim: int = 192
    history_size: int = 3
    projector_hidden_dim: int = 2048
    action_smoothed_dim: int = 10
    action_mlp_scale: int = 4
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_dim_head: int = 64
    predictor_mlp_dim: int = 2048
    predictor_dropout: float = 0.1
    predictor_emb_dropout: float = 0.0
    dtype: Any = jnp.bfloat16

    def setup(self):
        self.encoder = encoder_modules['impala_small']()
        self.projector = ProjectionMLP(self.embed_dim, hidden_dim=self.projector_hidden_dim, dtype=self.dtype)
        self.action_encoder = ActionEmbedder(
            self.embed_dim,
            smoothed_dim=self.action_smoothed_dim,
            mlp_scale=self.action_mlp_scale,
            dtype=self.dtype,
        )
        self.predictor = ARPredictor(
            num_frames=self.history_size,
            dim=self.embed_dim,
            depth=self.predictor_depth,
            heads=self.predictor_heads,
            dim_head=self.predictor_dim_head,
            mlp_dim=self.predictor_mlp_dim,
            dropout=self.predictor_dropout,
            emb_dropout=self.predictor_emb_dropout,
            dtype=self.dtype,
        )
        self.pred_projector = ProjectionMLP(self.embed_dim, hidden_dim=self.projector_hidden_dim, dtype=self.dtype)

    def encode_pixels(self, pixels, *, train):
        leading_shape = pixels.shape[:-3]
        flat_pixels = pixels.reshape(-1, *pixels.shape[-3:])
        embeddings = self.encoder(flat_pixels, train=train)
        embeddings = self.projector(embeddings, train=train)
        return embeddings.reshape(*leading_shape, self.embed_dim)

    def predict_embeddings(self, embeddings, actions, *, train):
        action_embeddings = self.action_encoder(actions)
        predictions = self.predictor(embeddings, action_embeddings, train=train)
        leading_shape = predictions.shape[:-1]
        predictions = self.pred_projector(predictions.reshape(-1, self.embed_dim), train=train)
        return predictions.reshape(*leading_shape, self.embed_dim)

    def __call__(self, pixels, actions, *, train):
        embeddings = self.encode_pixels(pixels, train=train)
        predictions = self.predict_embeddings(
            embeddings[:, : self.history_size],
            actions[:, : self.history_size],
            train=train,
        )
        return embeddings, predictions

    def _rollout_predictions(self, pixels, goals, action_candidates):
        """Return goal embeddings and all autoregressive rollout checkpoints."""
        num_samples = action_candidates.shape[1]
        context_embeddings = self.encode_pixels(pixels[:, 0], train=False)
        context_embeddings = jnp.broadcast_to(
            context_embeddings[:, None],
            (context_embeddings.shape[0], num_samples, *context_embeddings.shape[1:]),
        )
        goal_embeddings = self.encode_pixels(goals[:, 0], train=False)[..., -1, :]

        history = context_embeddings.shape[2]
        horizon = action_candidates.shape[2]
        if history > horizon:
            raise ValueError(f'Context length {history} exceeds planning horizon {horizon}.')
        actions = action_candidates[:, :, :history]
        future_actions = action_candidates[:, :, history:]
        embeddings = context_embeddings

        batch_size = embeddings.shape[0]
        rollout_predictions = []
        for step in range(horizon - history):
            flat_embeddings = embeddings.reshape(batch_size * num_samples, embeddings.shape[2], self.embed_dim)
            flat_actions = actions.reshape(batch_size * num_samples, actions.shape[2], actions.shape[3])
            prediction = self.predict_embeddings(
                flat_embeddings[:, -self.history_size :],
                flat_actions[:, -self.history_size :],
                train=False,
            )[:, -1]
            rollout_predictions.append(prediction.reshape(batch_size, num_samples, self.embed_dim))
            embeddings = jnp.concatenate(
                [embeddings, prediction.reshape(batch_size, num_samples, 1, self.embed_dim)],
                axis=2,
            )
            actions = jnp.concatenate([actions, future_actions[:, :, step : step + 1]], axis=2)

        flat_embeddings = embeddings.reshape(batch_size * num_samples, embeddings.shape[2], self.embed_dim)
        flat_actions = actions.reshape(batch_size * num_samples, actions.shape[2], actions.shape[3])
        prediction = self.predict_embeddings(
            flat_embeddings[:, -self.history_size :],
            flat_actions[:, -self.history_size :],
            train=False,
        )[:, -1]
        prediction = prediction.reshape(batch_size, num_samples, self.embed_dim)
        rollout_predictions.append(prediction)
        return goal_embeddings, jnp.stack(rollout_predictions, axis=2)

    def rollout_cost(self, pixels, goals, action_candidates):
        """Score candidates by final predicted-to-goal embedding distance."""
        goal_embeddings, predictions = self._rollout_predictions(pixels, goals, action_candidates)
        final_prediction = predictions[:, :, -1]
        return jnp.sum((final_prediction - goal_embeddings[:, None]) ** 2, axis=-1)

    def rollout_cost_min_over_horizon(self, pixels, goals, action_candidates):
        """Score by the closest predicted rollout checkpoint to the goal.

        With the evaluation runtime's single current frame and the default
        horizon/action block of 5/5, the checkpoints correspond to 5, 10, 15,
        20, and 25 future atomic actions.  The current observation is excluded.
        """
        goal_embeddings, predictions = self._rollout_predictions(pixels, goals, action_candidates)
        distances = jnp.sum((predictions - goal_embeddings[:, None, None]) ** 2, axis=-1)
        return jnp.min(distances, axis=-1)
