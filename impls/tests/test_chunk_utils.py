import unittest

import numpy as np
from utils.chunk_utils import compute_goal_conditioned_chunk_returns


class ChunkReturnsTest(unittest.TestCase):
    def test_negative_rewards_accumulate_over_complete_chunk(self):
        rewards, masks = compute_goal_conditioned_chunk_returns(
            np.asarray([[1, 2, 3]]), np.asarray([6]), discount=0.9, gc_negative=True
        )

        np.testing.assert_allclose(rewards, np.asarray([-(1.0 + 0.9 + 0.9**2)]), rtol=1e-6)
        np.testing.assert_array_equal(masks, np.asarray([1.0]))

    def test_goal_inside_chunk_stops_rewards_and_bootstrap(self):
        rewards, masks = compute_goal_conditioned_chunk_returns(
            np.asarray([[1, 2, 3]]), np.asarray([2]), discount=0.9, gc_negative=True
        )

        np.testing.assert_array_equal(rewards, np.asarray([-1.0]))
        np.testing.assert_array_equal(masks, np.asarray([0.0]))

    def test_positive_reward_is_counted_once_at_first_goal(self):
        rewards, masks = compute_goal_conditioned_chunk_returns(
            np.asarray([[1, 2, 2, 3]]), np.asarray([2]), discount=0.9, gc_negative=False
        )

        np.testing.assert_allclose(rewards, np.asarray([0.9]), rtol=1e-6)
        np.testing.assert_array_equal(masks, np.asarray([0.0]))

    def test_k_one_matches_atomic_reward_and_mask(self):
        rewards, masks = compute_goal_conditioned_chunk_returns(
            np.asarray([[1], [2]]), np.asarray([3, 2]), discount=0.9, gc_negative=True
        )

        np.testing.assert_array_equal(rewards, np.asarray([-1.0, 0.0]))
        np.testing.assert_array_equal(masks, np.asarray([1.0, 0.0]))


if __name__ == '__main__':
    unittest.main()
