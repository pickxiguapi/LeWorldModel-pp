"""LeWM sequence batches backed by official OGBench visual NPZ datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_split(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    archive = np.load(path)
    required = {'observations', 'actions', 'terminals'}
    missing = required.difference(archive.files)
    if missing:
        archive.close()
        raise ValueError(f'{path} is missing columns: {sorted(missing)}')
    observations = archive['observations']
    actions = archive['actions']
    terminals = archive['terminals'].astype(bool, copy=False)
    if not (len(observations) == len(actions) == len(terminals)):
        archive.close()
        raise ValueError(f'Row count mismatch in {path}.')
    if observations.ndim != 4 or observations.shape[-1] not in (1, 3, 4):
        archive.close()
        raise ValueError(
            f'Expected HWC visual observations in {path}, got {observations.shape}.'
        )
    return archive, observations, actions, terminals


def _clip_starts(terminals, span):
    ends = np.flatnonzero(terminals) + 1
    if not len(ends) or ends[-1] != len(terminals):
        ends = np.concatenate([ends, [len(terminals)]])
    offsets = np.concatenate([[0], ends[:-1]])
    starts = []
    for offset, end in zip(offsets, ends):
        length = int(end - offset)
        if length >= span:
            starts.append(offset + np.arange(length - span + 1, dtype=np.int64))
    if not starts:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(starts)


class LeWMNPZSequenceDataset:
    """Expose official train/validation NPZ files through the LeWM batch API."""

    def __init__(
        self,
        train_path,
        val_path,
        *,
        num_steps=4,
        frameskip=5,
        seed=3072,
        **_unused,
    ):
        if num_steps <= 0 or frameskip <= 0:
            raise ValueError('num_steps and frameskip must be positive.')
        self.num_steps = int(num_steps)
        self.frameskip = int(frameskip)
        self.span = self.num_steps * self.frameskip
        self._shuffle_rng = np.random.default_rng(seed)

        (
            self._train_archive,
            self._train_observations,
            self._train_actions,
            self._train_terminals,
        ) = _load_split(train_path)
        (
            self._val_archive,
            self._val_observations,
            self._val_actions,
            self._val_terminals,
        ) = _load_split(val_path)

        if self._train_observations.shape[1:] != self._val_observations.shape[1:]:
            raise ValueError('Train and validation observation shapes differ.')
        if self._train_actions.shape[1:] != self._val_actions.shape[1:]:
            raise ValueError('Train and validation action shapes differ.')

        self._train_starts = _clip_starts(self._train_terminals, self.span)
        self._val_starts = _clip_starts(self._val_terminals, self.span)
        if not len(self._train_starts) or not len(self._val_starts):
            raise ValueError(
                f'No complete clips for span={self.span}: '
                f'train={len(self._train_starts)} val={len(self._val_starts)}.'
            )
        self.train_indices = np.arange(len(self._train_starts), dtype=np.int64)
        self.val_indices = np.arange(
            len(self._train_starts),
            len(self._train_starts) + len(self._val_starts),
            dtype=np.int64,
        )

        valid_actions = self._train_actions[~self._train_terminals]
        valid_actions = valid_actions[~np.isnan(valid_actions).any(axis=1)]
        self.action_mean = valid_actions.mean(axis=0)
        self.action_std = valid_actions.std(axis=0, ddof=1)
        self.action_std = np.where(self.action_std > 0, self.action_std, 1.0)
        self.action_dim = int(valid_actions.shape[-1])
        self.observation_shape = tuple(self._train_observations.shape[1:])

    def close(self):
        self._train_archive.close()
        self._val_archive.close()

    def __len__(self):
        return len(self.train_indices) + len(self.val_indices)

    def shuffled_train_indices(self):
        return self._shuffle_rng.permutation(self.train_indices)

    def get_batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            raise ValueError('Cannot load an empty batch.')
        split_at = len(self._train_starts)
        if np.all(indices < split_at):
            starts = self._train_starts[indices]
            observations = self._train_observations
            actions = self._train_actions
        elif np.all(indices >= split_at):
            starts = self._val_starts[indices - split_at]
            observations = self._val_observations
            actions = self._val_actions
        else:
            raise ValueError('A batch cannot mix train and validation indices.')

        pixel_rows = (
            starts[:, None]
            + np.arange(self.num_steps, dtype=np.int64)[None, :] * self.frameskip
        )
        action_rows = starts[:, None] + np.arange(self.span, dtype=np.int64)[None, :]
        pixels = observations[pixel_rows]
        batch_actions = actions[action_rows]
        batch_actions = (batch_actions - self.action_mean) / self.action_std
        batch_actions = np.nan_to_num(batch_actions, nan=0.0, posinf=0.0, neginf=0.0)
        batch_actions = batch_actions.reshape(len(indices), self.num_steps, -1)
        return {
            'pixels': pixels.astype(np.uint8, copy=False),
            'action': batch_actions.astype(np.float32, copy=False),
        }
