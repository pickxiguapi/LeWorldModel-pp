"""Deterministic episode-level train/validation splits shared by all stages."""

from __future__ import annotations

import numpy as np


def split_episode_indices(num_episodes, train_fraction=0.9, seed=0):
    if num_episodes < 2:
        raise ValueError('At least two episodes are required for a train/validation split.')
    if not 0.0 < train_fraction < 1.0:
        raise ValueError('train_fraction must be in (0, 1).')
    permutation = np.random.default_rng(seed).permutation(num_episodes)
    train_count = int(np.floor(train_fraction * num_episodes))
    train_count = min(max(train_count, 1), num_episodes - 1)
    return permutation[:train_count], permutation[train_count:]
