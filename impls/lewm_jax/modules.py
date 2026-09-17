"""LeWM modules, preserving the reference PyTorch math and initialization."""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


def _torch_uniform(fan_in):
    return nn.initializers.uniform(scale=fan_in**-0.5)


class TorchLinear(nn.Module):
    """Linear with PyTorch ``nn.Linear`` default initialization."""

    features: int
    use_bias: bool = True
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        kernel = self.param('kernel', _torch_uniform(fan_in), (fan_in, self.features))
        y = jnp.matmul(x.astype(self.dtype), kernel.astype(self.dtype))
        if self.use_bias:
            bias = self.param('bias', _torch_uniform(fan_in), (self.features,))
            y = y + bias.astype(self.dtype)
        return y


class PyTorchBatchNorm1d(nn.Module):
    """BatchNorm1d with PyTorch momentum and unbiased running variance."""

    momentum: float = 0.1
    epsilon: float = 1e-5
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *, train):
        features = x.shape[-1]
        scale = self.param('scale', nn.initializers.ones, (features,))
        bias = self.param('bias', nn.initializers.zeros, (features,))
        running_mean = self.variable('batch_stats', 'mean', jnp.zeros, (features,), jnp.float32)
        running_var = self.variable('batch_stats', 'var', jnp.ones, (features,), jnp.float32)
        if train:
            x_float = x.astype(jnp.float32)
            axes = tuple(range(x.ndim - 1))
            mean = jnp.mean(x_float, axis=axes)
            variance = jnp.mean((x_float - mean) ** 2, axis=axes)
            sample_count = x.size // features
            unbiased = variance * sample_count / max(sample_count - 1, 1)
            running_mean.value = (1 - self.momentum) * running_mean.value + self.momentum * mean
            running_var.value = (1 - self.momentum) * running_var.value + self.momentum * unbiased
        else:
            mean, variance = running_mean.value, running_var.value
        normalized = (x.astype(jnp.float32) - mean) * jax.lax.rsqrt(variance + self.epsilon)
        return (normalized * scale + bias).astype(self.dtype)


class ProjectionMLP(nn.Module):
    dim: int = 192
    hidden_dim: int = 2048
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *, train):
        x = TorchLinear(self.hidden_dim, dtype=self.dtype, name='linear_1')(x)
        x = PyTorchBatchNorm1d(dtype=self.dtype, name='batch_norm')(x, train=train)
        x = nn.gelu(x, approximate=False)
        return TorchLinear(self.dim, dtype=self.dtype, name='linear_2')(x)


class ActionEmbedder(nn.Module):
    dim: int = 192
    smoothed_dim: int = 10
    mlp_scale: int = 4
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, actions):
        x = TorchLinear(self.smoothed_dim, dtype=self.dtype, name='patch_embed')(actions.astype(jnp.float32))
        x = TorchLinear(self.mlp_scale * self.dim, dtype=self.dtype, name='linear_1')(x)
        x = nn.silu(x)
        return TorchLinear(self.dim, dtype=self.dtype, name='linear_2')(x)


class PredictorAttention(nn.Module):
    dim: int = 192
    heads: int = 16
    dim_head: int = 64
    dropout: float = 0.1
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *, train):
        x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name='norm')(x)
        qkv = TorchLinear(self.heads * self.dim_head * 3, use_bias=False, dtype=self.dtype, name='to_qkv')(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def reshape(value):
            return value.reshape(value.shape[0], value.shape[1], self.heads, self.dim_head).transpose(0, 2, 1, 3)

        q, k, v = reshape(q), reshape(k), reshape(v)
        logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) * (self.dim_head**-0.5)
        causal = jnp.tril(jnp.ones((x.shape[1], x.shape[1]), dtype=bool))
        logits = jnp.where(causal[None, None], logits, jnp.finfo(jnp.float32).min)
        attention = nn.softmax(logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        attention = nn.Dropout(rate=self.dropout, name='attention_dropout')(attention, deterministic=not train)
        out = jnp.einsum('bhqk,bhkd->bhqd', attention, v)
        out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.heads * self.dim_head)
        out = TorchLinear(self.dim, dtype=self.dtype, name='to_out')(out)
        return nn.Dropout(rate=self.dropout, name='output_dropout')(out, deterministic=not train)


class PredictorFeedForward(nn.Module):
    dim: int = 192
    hidden_dim: int = 2048
    dropout: float = 0.1
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *, train):
        x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name='norm')(x)
        x = TorchLinear(self.hidden_dim, dtype=self.dtype, name='linear_1')(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dropout(rate=self.dropout, name='dropout_1')(x, deterministic=not train)
        x = TorchLinear(self.dim, dtype=self.dtype, name='linear_2')(x)
        return nn.Dropout(rate=self.dropout, name='dropout_2')(x, deterministic=not train)


class ConditionalBlock(nn.Module):
    dim: int = 192
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, condition, *, train):
        modulation = nn.Dense(
            6 * self.dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            name='adaLN_modulation',
        )(nn.silu(condition))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)
        y = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False, dtype=self.dtype, name='norm_1')(x)
        y = y * (1 + scale_msa) + shift_msa
        x = x + gate_msa * PredictorAttention(
            self.dim, self.heads, self.dim_head, self.dropout, self.dtype, name='attention'
        )(y, train=train)
        y = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False, dtype=self.dtype, name='norm_2')(x)
        y = y * (1 + scale_mlp) + shift_mlp
        return x + gate_mlp * PredictorFeedForward(
            self.dim, self.mlp_dim, self.dropout, self.dtype, name='feed_forward'
        )(y, train=train)


class ARPredictor(nn.Module):
    num_frames: int = 3
    dim: int = 192
    depth: int = 6
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1
    emb_dropout: float = 0.0
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, embeddings, action_embeddings, *, train):
        position = self.param('position_embedding', nn.initializers.normal(1.0), (1, self.num_frames, self.dim))
        x = embeddings + position[:, : embeddings.shape[1]].astype(self.dtype)
        x = nn.Dropout(rate=self.emb_dropout, name='embedding_dropout')(x, deterministic=not train)
        for index in range(self.depth):
            x = ConditionalBlock(
                self.dim, self.heads, self.dim_head, self.mlp_dim, self.dropout, self.dtype, name=f'block_{index}'
            )(x, action_embeddings, train=train)
        return nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name='final_norm')(x)
