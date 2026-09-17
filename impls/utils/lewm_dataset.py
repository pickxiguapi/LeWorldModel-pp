"""Lazy OGBench dataset view over LeWM's JPEG-backed Lance format."""

from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

try:
    import hdf5plugin  # noqa: F401  # Register the source HDF5 Blosc filter.
except ImportError as exc:  # pragma: no cover
    raise ImportError('LeWM datasets require hdf5plugin.') from exc


def _compact_boundary_arrays(size, final_rows):
    original_terminals = np.zeros(size, dtype=np.float32)
    original_terminals[final_rows] = 1.0
    valids = 1.0 - original_terminals
    next_terminals = np.concatenate([original_terminals[1:], [1.0]])
    terminals = np.minimum(original_terminals + next_terminals, 1.0)
    return terminals.astype(np.float32), valids


def _standardize_actions(actions, reference_actions):
    """Match eval_ff.py's dataset-wide StandardScaler convention."""
    mean = np.nanmean(reference_actions, axis=0, dtype=np.float64).astype(np.float32)
    std = np.nanstd(reference_actions, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 0, std, 1.0).astype(np.float32)
    normalized = (actions.astype(np.float32, copy=False) - mean) / std
    return np.nan_to_num(normalized, nan=0.0), mean, std


class LazyLancePixelArray:
    """Array-like random access to a JPEG pixel column in Lance."""

    def __init__(self, table, start, stop):
        from lancedb.permutation import Permutation

        self.start = int(start)
        self.stop = int(stop)
        self._permutation = Permutation.identity(table).select_columns(['pixels']).with_format('arrow')
        first = self._decode(self._fetch([self.start])[0])
        self.shape = (self.stop - self.start, *first.shape)
        self.dtype = first.dtype

    def __len__(self):
        return self.shape[0]

    def _fetch(self, rows):
        batch = self._permutation.__getitems__([int(row) for row in rows])
        index = batch.schema.get_field_index('pixels')
        return batch.column(index).to_pylist()

    @staticmethod
    def _decode(blob):
        with Image.open(io.BytesIO(blob)) as image:
            return np.asarray(image.convert('RGB'), dtype=np.uint8)

    def __getitem__(self, index):
        if isinstance(index, slice):
            begin, end, stride = index.indices(len(self))
            indices = np.arange(begin, end, stride, dtype=np.int64)
        else:
            indices = np.asarray(index)

        scalar = indices.ndim == 0
        flat = indices.reshape(-1).astype(np.int64, copy=False)
        flat = np.where(flat < 0, flat + len(self), flat)
        if np.any((flat < 0) | (flat >= len(self))):
            raise IndexError('Pixel index out of range.')

        absolute = flat + self.start
        unique, inverse = np.unique(absolute, return_inverse=True)
        decoded = np.stack([self._decode(blob) for blob in self._fetch(unique)])
        values = decoded[inverse].reshape((*indices.shape, *self.shape[1:]))
        return values[()] if scalar else values


class LeWMLanceDataset:
    """OGBench compact-dataset view over one contiguous Lance split."""

    lazy = True

    def __init__(self, lance_path, split, validation_fraction=0.05):
        import lancedb
        import pyarrow as pa

        path = Path(lance_path)
        if path.suffix != '.lance':
            raise ValueError('Lance path must point to a *.lance table directory.')
        table = lancedb.connect(str(path.parent)).open_table(path.stem)

        episode_chunks = []
        action_chunks = []
        reader = table.to_lance().scanner(columns=['episode_idx', 'action']).to_reader()
        for batch in reader:
            episode_col = batch.column(batch.schema.get_field_index('episode_idx'))
            episode_chunks.append(episode_col.to_numpy(zero_copy_only=False))
            action_col = batch.column(batch.schema.get_field_index('action'))
            if pa.types.is_fixed_size_list(action_col.type):
                actions = (
                    action_col.flatten()
                    .to_numpy(zero_copy_only=False)
                    .reshape(len(action_col), action_col.type.list_size)
                )
            else:
                actions = np.asarray(action_col.to_pylist(), dtype=np.float32)
            action_chunks.append(actions.astype(np.float32, copy=False))

        episode_ids = np.concatenate(episode_chunks).astype(np.int64, copy=False)
        all_actions = np.concatenate(action_chunks)
        changes = np.flatnonzero(np.diff(episode_ids) != 0) + 1
        episode_offsets = np.concatenate([[0], changes]).astype(np.int64)
        episode_lengths = np.diff(np.concatenate([episode_offsets, [len(episode_ids)]])).astype(np.int64)

        split_episode = int(len(episode_offsets) * (1 - validation_fraction))
        split_episode = min(max(split_episode, 1), len(episode_offsets) - 1)
        if split == 'train':
            selected_offsets = episode_offsets[:split_episode]
            selected_lengths = episode_lengths[:split_episode]
            start, stop = 0, int(episode_offsets[split_episode])
        elif split == 'val':
            selected_offsets = episode_offsets[split_episode:]
            selected_lengths = episode_lengths[split_episode:]
            start, stop = int(episode_offsets[split_episode]), len(episode_ids)
        else:
            raise ValueError(f'Unknown split: {split}')

        self.start = start
        self.stop = stop
        self.size = stop - start
        self.observations = LazyLancePixelArray(table, start, stop)

        # eval_ff.py fits StandardScaler on the original full HDF5 action
        # column. Use it here too; Cube's final HDF5 actions are NaN while the
        # converted Lance table contains unused zero placeholders there.
        source_hdf5 = path.with_suffix('.h5')
        if not source_hdf5.is_file():
            raise FileNotFoundError(f'Original HDF5 dataset is required for action statistics: {source_hdf5}')
        with h5py.File(source_hdf5, 'r') as h5_file:
            reference_actions = h5_file['action'][...].astype(np.float32, copy=False)
        self.actions, self.action_mean, self.action_std = _standardize_actions(
            all_actions[start:stop], reference_actions
        )

        final_rows = selected_offsets + selected_lengths - 1 - start
        self.terminals, self.valids = _compact_boundary_arrays(self.size, final_rows)
        (self.valid_idxs,) = np.nonzero(self.valids > 0)
        self._fields = {
            'observations': self.observations,
            'actions': self.actions,
            'terminals': self.terminals,
            'valids': self.valids,
        }

    def __len__(self):
        return self.size

    def __contains__(self, key):
        return key in self._fields

    def __getitem__(self, key):
        return self._fields[key]

    def get_random_idxs(self, num_idxs):
        positions = np.random.randint(len(self.valid_idxs), size=num_idxs)
        return self.valid_idxs[positions]

    def sample(self, batch_size, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        idxs = np.asarray(idxs)
        result = {key: value[idxs] for key, value in self._fields.items()}
        result['next_observations'] = self.observations[np.minimum(idxs + 1, self.size - 1)]
        return result


def make_lewm_lance_datasets(lance_path, validation_fraction=0.05):
    kwargs = {
        'lance_path': lance_path,
        'validation_fraction': validation_fraction,
    }
    return (
        LeWMLanceDataset(split='train', **kwargs),
        LeWMLanceDataset(split='val', **kwargs),
    )
