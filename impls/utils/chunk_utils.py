"""Pure NumPy utilities for temporally extended action transitions."""

import numpy as np


def compute_goal_conditioned_chunk_returns(chunk_state_idxs, goal_idxs, discount, gc_negative):
    """Compute discounted rewards and bootstrap masks for relabeled goals.

    Args:
        chunk_state_idxs: Integer array of shape ``(batch, chunk_size)`` for
            states ``s[t], ..., s[t + k - 1]``.
        goal_idxs: Integer array of shape ``(batch,)``.
        discount: Atomic-step discount factor.
        gc_negative: Use ``0 at goal, -1 otherwise`` when true and
            ``1 at goal, 0 otherwise`` when false.
    """
    chunk_state_idxs = np.asarray(chunk_state_idxs)
    goal_idxs = np.asarray(goal_idxs)
    if chunk_state_idxs.ndim != 2:
        raise ValueError(f'chunk_state_idxs must be rank 2, got shape {chunk_state_idxs.shape}.')
    if goal_idxs.shape != (chunk_state_idxs.shape[0],):
        raise ValueError(f'goal_idxs must have shape ({chunk_state_idxs.shape[0]},), got {goal_idxs.shape}.')

    successes = chunk_state_idxs == goal_idxs[:, None]
    atomic_masks = 1.0 - successes.astype(np.float32)
    alive = np.concatenate(
        [
            np.ones((chunk_state_idxs.shape[0], 1), dtype=np.float32),
            np.cumprod(atomic_masks[:, :-1], axis=1),
        ],
        axis=1,
    )
    atomic_rewards = successes.astype(np.float32) - float(gc_negative)
    discounts = np.float32(discount) ** np.arange(chunk_state_idxs.shape[1])
    rewards = np.sum(discounts[None, :] * alive * atomic_rewards, axis=1)
    masks = np.prod(atomic_masks, axis=1)
    return rewards.astype(np.float32), masks.astype(np.float32)
