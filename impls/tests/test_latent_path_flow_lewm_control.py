import unittest

import jax
import jax.numpy as jnp
import numpy as np
from latent_path_flow_lewm_control import (
    LatentPathFlow,
    sample_path,
    waypoint_steps,
)


class SubgoalGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.model = LatentPathFlow(
            embed_dim=8,
            num_waypoints=2,
            hidden_dim=16,
            depth=2,
            num_heads=4,
            ff_dim=32,
            time_dim=8,
            history_size=3,
        )
        self.history = jnp.zeros((2, 3, 8), dtype=jnp.float32)
        self.goals = jnp.ones((2, 8), dtype=jnp.float32)
        self.params = self.model.init(
            jax.random.PRNGKey(0),
            jnp.zeros((2, 2, 8), dtype=jnp.float32),
            self.history,
            self.goals,
            jnp.zeros((2,), dtype=jnp.float32),
        )['params']

    def test_waypoints_follow_action_chunks(self):
        self.assertEqual(waypoint_steps(10, 5), (5, 10))
        with self.assertRaisesRegex(ValueError, 'divisible'):
            waypoint_steps(9, 5)

    def test_sampling_is_deterministic_for_a_fixed_key(self):
        key = jax.random.PRNGKey(7)
        first = sample_path(self.model, self.params, self.history, self.goals, key, num_steps=2)
        second = sample_path(self.model, self.params, self.history, self.goals, key, num_steps=2)
        self.assertEqual(first.shape, (2, 2, 8))
        np.testing.assert_allclose(first, second)

if __name__ == '__main__':
    unittest.main()
