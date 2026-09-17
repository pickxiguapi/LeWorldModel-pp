"""Trajectory-aware utilities for frozen-LeWM latent subgoal training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LatentCache:
    z: np.ndarray
    episode_offsets: np.ndarray
    episode_lengths: np.ndarray
    metadata: dict


def load_latent_cache(path):
    """Load and validate one completed latent cache into host memory."""
    import h5py

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Latent cache not found: {path}')
    with h5py.File(path, 'r') as file:
        if file.attrs.get('format') != 'lewm_latent_dataset':
            raise ValueError(f'Not a LeWM latent dataset: {path}')
        if file.attrs.get('status') != 'complete':
            raise ValueError(f'Latent cache is not complete: {path}')
        z = np.asarray(file['z'], dtype=np.float32)
        offsets = np.asarray(file['ep_offset'], dtype=np.int64)
        lengths = np.asarray(file['ep_len'], dtype=np.int64)
        metadata = {key: value for key, value in file.attrs.items()}
    validate_trajectory_layout(len(z), offsets, lengths)
    embed_dim = int(metadata['embed_dim'])
    if z.ndim != 2 or z.shape[1] != embed_dim:
        raise ValueError(f'Invalid z shape {z.shape}; expected (*, {embed_dim}).')
    if not np.isfinite(z).all():
        raise FloatingPointError(f'Latent cache contains non-finite values: {path}')
    return LatentCache(z=z, episode_offsets=offsets, episode_lengths=lengths, metadata=metadata)


def validate_trajectory_layout(num_rows, offsets, lengths):
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if offsets.ndim != 1 or lengths.ndim != 1 or offsets.shape != lengths.shape:
        raise ValueError('ep_offset and ep_len must be same-shape one-dimensional arrays.')
    if not len(offsets) or offsets[0] != 0 or np.any(lengths <= 0):
        raise ValueError('Trajectory layout must be non-empty, start at zero, and have positive lengths.')
    expected_offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError('ep_offset is not contiguous with ep_len.')
    if int(lengths.sum()) != int(num_rows):
        raise ValueError(f'ep_len sums to {lengths.sum()}, but z has {num_rows} rows.')


def split_episodes(num_episodes, train_fraction=0.95, seed=0):
    if num_episodes < 2:
        raise ValueError('At least two episodes are required for a train/validation split.')
    if not 0.0 < train_fraction < 1.0:
        raise ValueError('train_fraction must be in (0, 1).')
    permutation = np.random.default_rng(seed).permutation(num_episodes)
    train_count = int(np.floor(train_fraction * num_episodes))
    train_count = min(max(train_count, 1), num_episodes - 1)
    return permutation[:train_count], permutation[train_count:]


def build_valid_transitions(
    offsets, lengths, episode_indices, min_future_steps=1
):
    """Return each t with enough future steps and its episode-final row."""
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if int(min_future_steps) <= 0:
        raise ValueError('min_future_steps must be positive.')
    current_rows = []
    final_rows = []
    for episode in np.asarray(episode_indices, dtype=np.int64):
        offset = int(offsets[episode])
        length = int(lengths[episode])
        if length <= int(min_future_steps):
            continue
        count = length - int(min_future_steps)
        current_rows.append(np.arange(offset, offset + count, dtype=np.int32))
        final_rows.append(np.full(count, offset + length - 1, dtype=np.int32))
    if not current_rows:
        raise ValueError('Selected episodes contain no valid transitions.')
    return np.concatenate(current_rows), np.concatenate(final_rows)


def build_history_indices(current_rows, episode_offsets, history_size):
    """Return episode-safe oldest-to-newest history rows for every current row."""
    current_rows = np.asarray(current_rows, dtype=np.int64)
    episode_offsets = np.asarray(episode_offsets, dtype=np.int64)
    if current_rows.ndim != 1 or episode_offsets.ndim != 1:
        raise ValueError('Current rows and episode offsets must be one-dimensional.')
    if not len(episode_offsets) or episode_offsets[0] != 0:
        raise ValueError('Episode offsets must be non-empty and start at zero.')
    if history_size <= 0:
        raise ValueError('History size must be positive.')
    if len(current_rows) == 0:
        return np.empty((0, history_size), dtype=np.int32)
    if np.any(current_rows < 0):
        raise ValueError('Current rows cannot be negative.')

    episode_ids = np.searchsorted(episode_offsets, current_rows, side='right') - 1
    episode_starts = episode_offsets[episode_ids]
    lags = np.arange(history_size - 1, -1, -1, dtype=np.int64)
    history = np.maximum(current_rows[:, None] - lags[None], episode_starts[:, None])
    return history.astype(np.int32)


def build_distance_balanced_transition_tables(
    offsets,
    lengths,
    episode_indices,
    goal_steps,
    history_size,
):
    """Build padded per-distance transition tables for distance-first sampling."""
    goal_steps = np.asarray(goal_steps, dtype=np.int32)
    if goal_steps.ndim != 1 or not len(goal_steps) or np.any(goal_steps <= 0):
        raise ValueError('goal_steps must be a non-empty vector of positive steps.')
    if len(np.unique(goal_steps)) != len(goal_steps):
        raise ValueError('goal_steps must not contain duplicates.')

    current_groups = []
    history_groups = []
    for goal_step in goal_steps:
        current, _ = build_valid_transitions(
            offsets,
            lengths,
            episode_indices,
            min_future_steps=int(goal_step),
        )
        current_groups.append(current)
        history_groups.append(
            build_history_indices(current, offsets, history_size)
        )

    counts = np.asarray([len(group) for group in current_groups], dtype=np.int32)
    max_count = int(counts.max())
    current_table = np.empty((len(goal_steps), max_count), dtype=np.int32)
    history_table = np.empty(
        (len(goal_steps), max_count, history_size), dtype=np.int32
    )
    for index, (current, history) in enumerate(
        zip(current_groups, history_groups)
    ):
        count = len(current)
        current_table[index, :count] = current
        history_table[index, :count] = history
        current_table[index, count:] = current[0]
        history_table[index, count:] = history[0]
    return current_table, history_table, counts


def sample_distance_balanced_future_pairs(
    current_table,
    counts,
    goal_steps,
    num_pairs,
    subgoal_steps,
    seed,
):
    """Sample nearly equal validation counts for every configured goal distance."""
    current_table = np.asarray(current_table, dtype=np.int32)
    counts = np.asarray(counts, dtype=np.int32)
    goal_steps = np.asarray(goal_steps, dtype=np.int32)
    if current_table.ndim != 2:
        raise ValueError('current_table must be two-dimensional.')
    if counts.shape != (len(current_table),) or goal_steps.shape != counts.shape:
        raise ValueError('counts and goal_steps must match current_table rows.')
    if np.any(counts <= 0) or np.any(counts > current_table.shape[1]):
        raise ValueError('counts must identify non-empty rows within current_table.')
    if num_pairs <= 0 or subgoal_steps <= 0:
        raise ValueError('num_pairs and subgoal_steps must be positive.')

    rng = np.random.default_rng(seed)
    distance_indices = np.arange(num_pairs, dtype=np.int32) % len(goal_steps)
    rng.shuffle(distance_indices)
    positions = np.floor(
        rng.random(num_pairs) * counts[distance_indices]
    ).astype(np.int32)
    t = current_table[distance_indices, positions]
    g = t + goal_steps[distance_indices]
    target = np.minimum(t + int(subgoal_steps), g).astype(np.int32)
    return t, g, target


def sample_future_pairs(
    valid_t,
    final_t,
    num_pairs,
    subgoal_steps,
    seed,
    goal_stride=1,
):
    """Create fixed same-trajectory future-goal pairs for validation."""
    valid_t = np.asarray(valid_t, dtype=np.int32)
    final_t = np.asarray(final_t, dtype=np.int32)
    if valid_t.shape != final_t.shape or valid_t.ndim != 1:
        raise ValueError('valid_t and final_t must be same-shape one-dimensional arrays.')
    if num_pairs <= 0 or subgoal_steps <= 0 or goal_stride <= 0:
        raise ValueError('num_pairs, subgoal_steps, and goal_stride must be positive.')
    rng = np.random.default_rng(seed)
    positions = rng.integers(len(valid_t), size=num_pairs)
    t = valid_t[positions]
    episode_end = final_t[positions]
    distances = rng.random(num_pairs)
    future_counts = (episode_end - t) // int(goal_stride)
    if np.any(future_counts <= 0):
        raise ValueError('Every sampled current row must have one aligned future goal.')
    goal_blocks = 1 + np.floor(distances * future_counts).astype(np.int32)
    g = t + goal_blocks * int(goal_stride)
    target = np.minimum(t + int(subgoal_steps), g).astype(np.int32)
    return t, g, target


def build_fixed_offset_validation_manifest(
    offsets,
    lengths,
    episode_indices,
    *,
    num_pairs,
    history_size,
    goal_offset,
    subgoal_steps,
    action_block,
    seed,
):
    """Build a deterministic held-out manifest with an exact goal offset."""
    if num_pairs <= 0 or history_size <= 0:
        raise ValueError('num_pairs and history_size must be positive.')
    if goal_offset < subgoal_steps or subgoal_steps <= 0 or action_block <= 0:
        raise ValueError(
            'goal_offset must cover positive subgoal_steps and action_block.'
        )
    if subgoal_steps % action_block:
        raise ValueError('subgoal_steps must be divisible by action_block.')

    valid_t, _ = build_valid_transitions(
        offsets,
        lengths,
        episode_indices,
        min_future_steps=goal_offset,
    )
    rng = np.random.default_rng(seed)
    positions = rng.integers(len(valid_t), size=num_pairs)
    current = valid_t[positions].astype(np.int32)
    history = build_history_indices(current, offsets, history_size)
    goal = (current + int(goal_offset)).astype(np.int32)
    waypoint_steps = np.arange(
        action_block, subgoal_steps + 1, action_block, dtype=np.int32
    )
    waypoint_targets = current[:, None] + waypoint_steps[None]
    episode_ids = (
        np.searchsorted(np.asarray(offsets), current, side='right') - 1
    ).astype(np.int32)
    return {
        'episode_ids': episode_ids,
        'current_indices': current,
        'history_indices': history.astype(np.int32),
        'goal_indices': goal,
        'waypoint_steps': waypoint_steps,
        'waypoint_target_indices': waypoint_targets.astype(np.int32),
        'endpoint_target_indices': waypoint_targets[:, -1].astype(np.int32),
    }
