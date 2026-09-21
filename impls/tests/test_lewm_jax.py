import jax
import jax.numpy as jnp
from lewm_jax import ARCHITECTURE, LeWM, lewm_loss
from train_lewm_control import LeWMConfig


def test_training_defaults_match_lewm_with_impala_encoder():
    config = LeWMConfig()
    assert config.architecture == ARCHITECTURE
    assert config.encoder == 'impala_small'
    assert config.epochs == 10
    assert config.batch_size == 128
    assert config.history_size == 3
    assert config.frameskip == 5
    assert config.predictor_depth == 6
    assert config.predictor_heads == 16
    assert config.learning_rate == 5e-5
    assert config.weight_decay == 1e-3
    assert config.sigreg_weight == 0.09


def test_impala_lewm_forward_and_loss():
    model = LeWM(image_size=64, dtype=jnp.float32)
    pixels = jnp.zeros((2, 4, 64, 64, 3), dtype=jnp.uint8)
    actions = jnp.zeros((2, 4, 10), dtype=jnp.float32)
    params_key, dropout_key, sigreg_key = jax.random.split(jax.random.PRNGKey(0), 3)
    variables = model.init({'params': params_key, 'dropout': dropout_key}, pixels, actions, train=False)
    embeddings, predictions = model.apply(variables, pixels, actions, train=False, rngs={'dropout': dropout_key})
    loss, (metrics, batch_stats) = lewm_loss(
        model,
        variables,
        {'pixels': pixels, 'action': actions},
        train=True,
        dropout_key=dropout_key,
        sigreg_key=sigreg_key,
        sigreg_num_proj=8,
    )
    assert embeddings.shape == (2, 4, 192)
    assert predictions.shape == (2, 3, 192)
    assert jnp.isfinite(loss)
    assert all(jnp.isfinite(value) for value in metrics.values())
    assert jax.tree_util.tree_leaves(batch_stats)


def test_rollout_min_over_horizon_cost_is_bounded_by_terminal_cost():
    model = LeWM(image_size=64, dtype=jnp.float32)
    pixels = jnp.zeros((2, 1, 1, 64, 64, 3), dtype=jnp.uint8)
    goals = jnp.zeros_like(pixels)
    candidates = jnp.zeros((2, 4, 5, 10), dtype=jnp.float32)
    variables = model.init(
        {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)},
        pixels[:, 0],
        jnp.zeros((2, 1, 10), dtype=jnp.float32),
        train=False,
    )

    terminal = model.apply(variables, pixels, goals, candidates, method=model.rollout_cost)
    minimum = model.apply(
        variables,
        pixels,
        goals,
        candidates,
        method=model.rollout_cost_min_over_horizon,
    )

    assert terminal.shape == (2, 4)
    assert minimum.shape == (2, 4)
    assert jnp.all(jnp.isfinite(terminal))
    assert jnp.all(jnp.isfinite(minimum))
    assert jnp.all(minimum <= terminal)
