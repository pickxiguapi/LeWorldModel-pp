"""LeWM sequence batches read lazily from JPEG-backed Lance tables.

This module implements the LeWM sequence sampling convention directly:
four observations are sampled five environment steps apart, while the
five intervening actions are flattened into one action chunk per observation.
The train/validation split is over clip indices (not episodes), matching
``torch.utils.data.random_split`` in the reference LeWM implementation.
"""

from __future__ import annotations

import io
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _fixed_list_to_numpy(column):
    """Convert a Lance/Arrow action column to a dense float32 array."""
    import pyarrow as pa

    if pa.types.is_fixed_size_list(column.type):
        return (
            column.flatten()
            .to_numpy(zero_copy_only=False)
            .reshape(len(column), column.type.list_size)
            .astype(np.float32, copy=False)
        )
    return np.asarray(column.to_pylist(), dtype=np.float32)


class LeWMSequenceDataset:
    """Random-access LeWM clips backed by one Lance table.

    The returned tensors have shapes ``pixels=(B, 4, H, W, 3)`` and
    ``action=(B, 4, 5 * action_dim)`` for the reference defaults.
    """

    def __init__(
        self,
        lance_path,
        *,
        num_steps=4,
        frameskip=5,
        train_fraction=0.9,
        seed=3072,
        decode_workers=6,
        normalize_pixels=False,
    ):
        import lancedb
        from lancedb.permutation import Permutation

        self.path = Path(lance_path)
        if self.path.suffix != '.lance' or not self.path.is_dir():
            raise ValueError(f'Expected a *.lance table directory, got {self.path}.')
        if num_steps <= 0 or frameskip <= 0:
            raise ValueError('num_steps and frameskip must be positive.')

        self.num_steps = int(num_steps)
        self.frameskip = int(frameskip)
        self.span = self.num_steps * self.frameskip
        self.normalize_pixels = bool(normalize_pixels)
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(decode_workers)))

        table = lancedb.connect(str(self.path.parent)).open_table(self.path.stem)
        self._pixel_rows = Permutation.identity(table).select_columns(['pixels']).with_format('arrow')
        self._action_rows = Permutation.identity(table).select_columns(['action']).with_format('arrow')

        episode_chunks = []
        action_chunks = []
        reader = table.to_lance().scanner(columns=['episode_idx', 'action']).to_reader()
        for batch in reader:
            episode_chunks.append(
                batch.column(batch.schema.get_field_index('episode_idx')).to_numpy(zero_copy_only=False)
            )
            action_chunks.append(_fixed_list_to_numpy(batch.column(batch.schema.get_field_index('action'))))

        episode_ids = np.concatenate(episode_chunks).astype(np.int64, copy=False)
        lance_actions = np.concatenate(action_chunks).astype(np.float32, copy=False)
        changes = np.flatnonzero(np.diff(episode_ids) != 0) + 1
        offsets = np.concatenate([[0], changes]).astype(np.int64)
        lengths = np.diff(np.concatenate([offsets, [len(episode_ids)]])).astype(np.int64)

        clip_starts = []
        for offset, length in zip(offsets, lengths):
            if length >= self.span:
                clip_starts.extend(offset + np.arange(length - self.span + 1, dtype=np.int64))
        self.clip_starts = np.asarray(clip_starts, dtype=np.int64)
        if not len(self.clip_starts):
            raise ValueError(f'No episode in {self.path} is long enough for span={self.span}.')

        # The reference loader caches non-pixel columns in memory. Prefer the
        # sibling HDF5 action column as both the batch source and statistics
        # source: Lance stores action vectors as float32 and replaces terminal
        # NaNs, whereas Reacher is float64 in the source HDF5 and the reference
        # computes its statistics before the final `.float()` conversion.
        source_hdf5 = self.path.with_suffix('.h5')
        if source_hdf5.is_file():
            import h5py

            try:
                import hdf5plugin  # noqa: F401
            except ImportError:
                pass
            with h5py.File(source_hdf5, 'r') as h5_file:
                self._source_actions = h5_file['action'][...]
            if len(self._source_actions) != len(episode_ids):
                raise ValueError(
                    f'Action row mismatch: {source_hdf5} has {len(self._source_actions)}, '
                    f'but {self.path} has {len(episode_ids)} rows.'
                )
            stats_actions = self._source_actions
        else:
            warnings.warn(
                f'{source_hdf5} is absent; action statistics omit episode-final Lance rows as a fallback.',
                stacklevel=2,
            )
            self._source_actions = None
            valid_action_rows = np.ones(len(lance_actions), dtype=bool)
            valid_action_rows[offsets + lengths - 1] = False
            stats_actions = lance_actions[valid_action_rows]
        stats_actions = stats_actions[~np.isnan(stats_actions).any(axis=1)]
        # Preserve the HDF5 source dtype while fitting statistics. This matters
        # most for Reacher, whose source action column is float64. The result is
        # converted to float32 only after normalization, as in LeWM.
        self.action_mean = stats_actions.mean(axis=0)
        self.action_std = stats_actions.std(axis=0, ddof=1)
        self.action_std = np.where(self.action_std > 0, self.action_std, 1.0)
        self.action_dim = int(stats_actions.shape[-1])

        # One seeded RNG drives both the split and all epoch shuffles. NumPy is
        # used here so the JAX training environment does not require PyTorch;
        # the split ratio and deterministic semantics match the reference,
        # although the exact permutation is backend-specific.
        self._shuffle_rng = np.random.default_rng(seed)
        permutation = self._shuffle_rng.permutation(len(self.clip_starts))
        train_size = math.floor(train_fraction * len(permutation))
        val_size = math.floor((1 - train_fraction) * len(permutation))
        # stable_pretraining.data.random_split distributes a fractional
        # remainder from the first split onward; with two splits this gives
        # the possible single extra clip to training.
        if train_size + val_size < len(permutation):
            train_size += 1
        self.train_indices = permutation[:train_size]
        self.val_indices = permutation[train_size:]

    def close(self):
        self._executor.shutdown(wait=True)

    def __len__(self):
        return len(self.clip_starts)

    @staticmethod
    def _decode(blob):
        with Image.open(io.BytesIO(blob)) as image:
            return np.asarray(image.convert('RGB'), dtype=np.uint8)

    def _fetch_pixels(self, absolute_rows):
        unique_rows, inverse = np.unique(absolute_rows, return_inverse=True)
        batch = self._pixel_rows.__getitems__(unique_rows.tolist())
        blobs = batch.column(batch.schema.get_field_index('pixels')).to_pylist()
        decoded = np.stack(list(self._executor.map(self._decode, blobs)))
        return decoded[inverse]

    def _fetch_actions(self, absolute_rows):
        if self._source_actions is not None:
            return self._source_actions[absolute_rows]
        unique_rows, inverse = np.unique(absolute_rows, return_inverse=True)
        batch = self._action_rows.__getitems__(unique_rows.tolist())
        actions = _fixed_list_to_numpy(batch.column(batch.schema.get_field_index('action')))
        return actions[inverse]

    def get_batch(self, indices):
        """Load dataset clip indices as one normalized NumPy batch."""
        indices = np.asarray(indices, dtype=np.int64)
        starts = self.clip_starts[indices]

        pixel_rows = starts[:, None] + np.arange(self.num_steps, dtype=np.int64)[None, :] * self.frameskip
        pixels = self._fetch_pixels(pixel_rows.reshape(-1)).reshape(
            len(indices), self.num_steps, *self._fetch_image_shape(pixel_rows[0, 0])
        )
        if self.normalize_pixels:
            pixels = pixels.astype(np.float32) / 255.0
            pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
        action_rows = starts[:, None] + np.arange(self.span, dtype=np.int64)[None, :]
        actions = self._fetch_actions(action_rows.reshape(-1)).reshape(len(indices), self.span, self.action_dim)
        actions = (actions - self.action_mean) / self.action_std
        actions = np.nan_to_num(actions, nan=0.0).reshape(len(indices), self.num_steps, -1)
        actions = actions.astype(np.float32, copy=False)
        return {'pixels': pixels, 'action': actions}

    def shuffled_train_indices(self):
        """Return the next deterministic epoch permutation."""
        order = self._shuffle_rng.permutation(len(self.train_indices))
        return self.train_indices[order]

    def _fetch_image_shape(self, absolute_row):
        # Fetching a shape through Lance for every batch would be wasteful. Cache
        # it after the first decoded frame while keeping construction lightweight.
        if not hasattr(self, '_image_shape'):
            batch = self._pixel_rows.__getitems__([int(absolute_row)])
            blob = batch.column(batch.schema.get_field_index('pixels')).to_pylist()[0]
            self._image_shape = self._decode(blob).shape
        return self._image_shape
