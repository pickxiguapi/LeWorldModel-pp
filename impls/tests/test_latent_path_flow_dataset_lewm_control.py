import numpy as np
from utils.latent_path_flow_dataset_lewm_control import (
    build_distance_balanced_transition_tables,
    build_fixed_offset_validation_manifest,
    build_history_indices,
    build_valid_transitions,
    sample_distance_balanced_future_pairs,
    sample_future_pairs,
    split_episodes,
    validate_trajectory_layout,
)


def test_layout_split_and_valid_transitions_are_episode_safe():
    offsets = np.asarray([0, 4, 7])
    lengths = np.asarray([4, 3, 5])
    validate_trajectory_layout(12, offsets, lengths)
    train_episodes, val_episodes = split_episodes(3, train_fraction=2 / 3, seed=7)
    assert set(train_episodes).isdisjoint(set(val_episodes))
    assert set(np.concatenate((train_episodes, val_episodes))) == {0, 1, 2}

    current, final = build_valid_transitions(offsets, lengths, [0, 2])
    np.testing.assert_array_equal(current, [0, 1, 2, 7, 8, 9, 10])
    np.testing.assert_array_equal(final, [3, 3, 3, 11, 11, 11, 11])


def test_hiql_future_sampling_and_k10_targets():
    valid_t = np.asarray([0, 10, 20], dtype=np.int32)
    final_t = np.asarray([30, 30, 30], dtype=np.int32)
    t, g, target = sample_future_pairs(valid_t, final_t, 10000, subgoal_steps=10, seed=3)
    assert np.all(g > t)
    assert np.all(g <= 30)
    np.testing.assert_array_equal(target, np.minimum(t + 10, g))
    assert np.any(g - t < 10)
    assert np.any(g - t > 10)
    np.testing.assert_array_equal(target[g - t <= 10], g[g - t <= 10])


def test_future_sampling_aligns_goal_distance_to_stride():
    valid_t = np.asarray([0, 1, 7], dtype=np.int32)
    final_t = np.asarray([30, 30, 30], dtype=np.int32)
    t, g, target = sample_future_pairs(
        valid_t,
        final_t,
        10_000,
        subgoal_steps=10,
        seed=5,
        goal_stride=5,
    )
    assert np.all(g > t)
    assert np.all(g <= 30)
    assert np.all((g - t) % 5 == 0)
    np.testing.assert_array_equal(target, np.minimum(t + 10, g))


def test_valid_transitions_can_require_one_full_goal_stride():
    current, final = build_valid_transitions(
        np.asarray([0]),
        np.asarray([8]),
        [0],
        min_future_steps=5,
    )
    np.testing.assert_array_equal(current, [0, 1, 2])
    np.testing.assert_array_equal(final, [7, 7, 7])


def test_three_frame_histories_repeat_pad_without_crossing_episodes():
    offsets = np.asarray([0, 4, 7])
    current = np.asarray([0, 1, 2, 3, 7, 8, 10], dtype=np.int32)
    history = build_history_indices(current, offsets, history_size=3)
    np.testing.assert_array_equal(
        history,
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [7, 7, 7],
            [7, 7, 8],
            [8, 9, 10],
        ],
    )


def test_future_goal_sampling_is_uniform_including_endpoints():
    valid_t = np.asarray([0], dtype=np.int32)
    final_t = np.asarray([4], dtype=np.int32)
    _, goals, _ = sample_future_pairs(valid_t, final_t, 100_000, subgoal_steps=10, seed=11)
    counts = np.bincount(goals, minlength=5)[1:]
    assert np.max(np.abs(counts - counts.mean())) < 0.02 * counts.mean()


def test_distance_first_sampling_balances_aligned_goals_up_to_25():
    offsets = np.asarray([0, 31, 67])
    lengths = np.asarray([31, 36])
    goal_steps = np.asarray([5, 10, 15, 20, 25])
    current, history, counts = build_distance_balanced_transition_tables(
        offsets,
        lengths,
        [0, 1],
        goal_steps,
        history_size=3,
    )
    assert current.shape[0] == len(goal_steps)
    assert history.shape == (*current.shape, 3)
    np.testing.assert_array_equal(counts, [57, 47, 37, 27, 17])

    t, g, target = sample_distance_balanced_future_pairs(
        current,
        counts,
        goal_steps,
        num_pairs=10_000,
        subgoal_steps=10,
        seed=17,
    )
    distances = g - t
    np.testing.assert_array_equal(np.unique(distances), goal_steps)
    np.testing.assert_array_equal(
        [np.count_nonzero(distances == step) for step in goal_steps],
        np.full(len(goal_steps), 2_000),
    )
    np.testing.assert_array_equal(target, np.minimum(t + 10, g))
    np.testing.assert_array_equal(target[distances == 5], g[distances == 5])


def test_fixed_offset_manifest_is_deterministic_and_episode_safe():
    offsets = np.asarray([0, 61, 132])
    lengths = np.asarray([61, 71, 81])
    kwargs = dict(
        num_pairs=100,
        history_size=3,
        goal_offset=50,
        subgoal_steps=10,
        action_block=5,
        seed=9,
    )
    first = build_fixed_offset_validation_manifest(offsets, lengths, [1, 2], **kwargs)
    repeated = build_fixed_offset_validation_manifest(offsets, lengths, [1, 2], **kwargs)
    for name in first:
        np.testing.assert_array_equal(first[name], repeated[name])
    np.testing.assert_array_equal(first['goal_indices'] - first['current_indices'], 50)
    np.testing.assert_array_equal(first['waypoint_steps'], [5, 10])
    np.testing.assert_array_equal(
        first['waypoint_target_indices'],
        first['current_indices'][:, None] + np.asarray([5, 10]),
    )
    assert set(np.unique(first['episode_ids'])) <= {1, 2}
