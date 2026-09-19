import ast
import unittest
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from latent_path_flow_runtime_lewm_control import SubgoalGenerator
from lewm_jax.planner_lewm_control import (
    LeWMPPController,
    _set_cem_anchors,
    reduce_rollout_costs,
    subgoal_planning_horizon,
)


class FakePrior:
    action_horizon = 5
    lewm_checkpoint = '/tmp/lewm.msgpack'

    def sample_actions(self, observations, goals, seed, temperature):
        del observations, seed
        assert temperature == 0.0
        np.testing.assert_array_equal(goals, np.full((1, 4, 4, 3), 9, dtype=np.uint8))
        return jnp.arange(10, dtype=jnp.float32)[None]


class FakePopulationPrior:
    action_horizon = 5

    def __init__(self):
        self.batch_size = None

    def sample_actions(self, observations, goals, seed, temperature):
        del goals, seed
        self.batch_size = observations.shape[0]
        assert temperature == 1.0
        return np.arange(self.batch_size * 10, dtype=np.float32).reshape(self.batch_size, 10)


class FakeWorldModel:
    def _rollout_predictions(self):
        raise AssertionError('Fake apply should receive, but not call, this method.')

    def apply(self, variables, pixels, goals, candidates, method=None):
        del variables, pixels, goals, method
        predictions = jnp.mean(candidates, axis=-1, keepdims=True)
        return jnp.zeros((1, 1), dtype=jnp.float32), predictions


class ControllerTest(unittest.TestCase):
    def test_public_planner_surface_has_no_legacy_controller(self):
        path = Path(__file__).parents[1] / 'lewm_jax' / 'planner_lewm_control.py'
        tree = ast.parse(path.read_text())
        public = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith('_')
        }
        self.assertEqual(
            public,
            {'subgoal_planning_horizon', 'reduce_rollout_costs', 'LeWMPPController'},
        )

    def test_horizon_and_temporal_costs(self):
        self.assertEqual(subgoal_planning_horizon(10, 5), 2)
        distances = jnp.asarray([[5.0, 1.0, 4.0], [2.0, 3.0, 6.0]])
        np.testing.assert_array_equal(reduce_rollout_costs(distances, 'last'), [4, 6])
        np.testing.assert_array_equal(reduce_rollout_costs(distances, 'moh'), [1, 2])
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            reduce_rollout_costs(distances, 'path_mean')

    def test_action_prior_initializes_only_the_first_block(self):
        controller = object.__new__(LeWMPPController)
        controller.horizon = 3
        controller.block_action_dim = 10
        controller.action_prior = FakePrior()
        controller.warm_starts = [None]
        pixels = np.zeros((1, 4, 4, 3), dtype=np.uint8)
        goals = np.full((1, 4, 4, 3), 9, dtype=np.uint8)
        mean = controller._initial_mean(0, pixels, goals, jax.random.PRNGKey(0))
        np.testing.assert_array_equal(mean[0], np.arange(10, dtype=np.float32))
        np.testing.assert_array_equal(mean[1:], np.zeros((2, 10), dtype=np.float32))

    def test_unexecuted_plan_suffix_warm_starts_the_next_plan(self):
        controller = object.__new__(LeWMPPController)
        controller.horizon = 3
        controller.block_action_dim = 2
        controller.action_prior = None
        controller.warm_starts = [np.asarray([[3.0, 4.0], [5.0, 6.0]])]
        mean = controller._initial_mean(
            0,
            np.zeros((1, 4, 4, 3), dtype=np.uint8),
            np.zeros((1, 4, 4, 3), dtype=np.uint8),
            None,
        )
        np.testing.assert_array_equal(mean, [[3.0, 4.0], [5.0, 6.0], [0.0, 0.0]])

    def test_policy_random_mixture_builds_fresh_population_for_each_iteration(self):
        controller = object.__new__(LeWMPPController)
        controller.action_prior = FakePopulationPrior()
        controller.action_prior_population_size = 4
        controller.iterations = 2
        controller.block_action_dim = 10
        pixels = np.zeros((1, 4, 4, 3), dtype=np.uint8)

        blocks = controller._action_prior_population(pixels, pixels, jax.random.PRNGKey(0))

        self.assertEqual(blocks.shape, (2, 4, 10))
        self.assertEqual(controller.action_prior.batch_size, 8)
        np.testing.assert_array_equal(blocks[0].reshape(-1), np.arange(40, dtype=np.float32))
        np.testing.assert_array_equal(blocks[1].reshape(-1), np.arange(40, 80, dtype=np.float32))

    def test_policy_best_of_n_scores_only_one_action_block(self):
        controller = object.__new__(LeWMPPController)
        controller.model = FakeWorldModel()
        controller.variables = {}
        controller.num_samples = 4
        controller.iterations = 1
        controller.topk = 2
        controller.var_scale = 1.0
        controller.cost_mode = 'moh'
        controller.subgoal_generator = object()
        controller.action_prior_mode = 'policy_best_of_n'
        controller.action_prior_population_size = 4
        controller.planner_action_low = None
        controller.planner_action_high = None
        controller.action_block = 2
        plan_one = jax.jit(controller._build_plan_one())
        proposals = jnp.asarray([[[3.0, 1.0], [2.0, 2.0], [0.0, 0.0], [-4.0, -2.0]]])

        selected = plan_one(
            jax.random.PRNGKey(0),
            jnp.zeros((1, 4, 4, 3), dtype=jnp.uint8),
            jnp.zeros((1, 4, 4, 3), dtype=jnp.uint8),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.float32),
            proposals,
        )

        self.assertEqual(selected.shape, (1, 2))
        np.testing.assert_array_equal(selected[0], proposals[0, 2])

    def test_runtime_uses_consecutive_observation_history(self):
        generator = object.__new__(SubgoalGenerator)
        generator.history_size = 3
        generator.histories = [
            deque(
                [
                    np.full((2, 2, 1), 1.0, dtype=np.float32),
                    np.full((2, 2, 1), 2.0, dtype=np.float32),
                    np.full((2, 2, 1), 3.0, dtype=np.float32),
                ],
                maxlen=3,
            )
        ]
        generator.generation_counts = np.zeros(1, dtype=np.int64)
        generator.embed_dim = 3
        generator.path_length = 2
        generator.seed = 42
        generator.encode_pixels = lambda pixels: np.repeat(
            np.asarray(pixels).mean(axis=(1, 2, 3))[:, None], 3, axis=1
        ).astype(np.float32)
        captured = {}

        def predict(history, goal, rng):
            del goal, rng
            captured['history'] = np.asarray(history)
            return jnp.repeat(history[:, -1:, :], 2, axis=1)

        generator._predict = predict
        path = generator.predict_path(0, np.full((2, 2, 1), 4.0, dtype=np.float32))
        np.testing.assert_array_equal(captured['history'][0, :, 0], [1, 2, 3])
        self.assertEqual(path.shape, (2, 3))

    def test_policy_mode_anchor_is_preserved_as_second_candidate(self):
        candidates = jnp.zeros((4, 2, 3), dtype=jnp.float32)
        mean = jnp.ones((2, 3), dtype=jnp.float32)
        policy = jnp.full((2, 3), 7.0, dtype=jnp.float32)
        mode = _set_cem_anchors(candidates, mean, policy, False)
        anchor = _set_cem_anchors(candidates, mean, policy, True)
        np.testing.assert_array_equal(mode[0], mean)
        np.testing.assert_array_equal(mode[1], 0)
        np.testing.assert_array_equal(anchor[0], mean)
        np.testing.assert_array_equal(anchor[1], policy)


if __name__ == '__main__':
    unittest.main()
