"""Precompute one frozen LeWM embedding per row of a JPEG-backed Lance dataset.

The output is a checkpoint-bound HDF5 cache.  It contains a float latent
matrix named ``z`` together with the source dataset's non-pixel HDF5 fields
and canonical trajectory indexing arrays.  Images are deliberately decoded
from Lance so that the encoder sees the same JPEG representation used during
LeWM training.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

FORMAT_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('cube', 'pusht', 'reacher', 'tworoom'), required=True)
    parser.add_argument('--lance-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--decode-workers', type=int, default=12)
    parser.add_argument('--output-dtype', choices=('float32', 'float16'), default='float32')
    parser.add_argument('--flush-every-batches', type=int, default=20)
    parser.add_argument('--log-every-batches', type=int, default=20)
    parser.add_argument(
        '--smoke-rows',
        type=int,
        default=0,
        help='Encode this many leading rows and exit without creating a cache.',
    )
    return parser.parse_args()


def sha256_file(path, block_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def compute_episode_layout(episode_ids):
    episode_ids = np.asarray(episode_ids)
    if episode_ids.ndim != 1 or not len(episode_ids):
        raise ValueError('episode_idx must be a non-empty one-dimensional array.')
    changes = np.flatnonzero(np.diff(episode_ids) != 0) + 1
    offsets = np.concatenate(([0], changes)).astype(np.int64, copy=False)
    lengths = np.diff(np.concatenate((offsets, [len(episode_ids)]))).astype(np.int64, copy=False)
    return offsets, lengths


def _copy_attrs(source, destination):
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def copy_non_pixel_hdf5(source_path, destination):
    """Copy every HDF5 field except image datasets into ``destination``."""
    import h5py

    try:
        import hdf5plugin  # noqa: F401  # Register optional source compression filters.
    except ImportError:
        pass

    source_path = Path(source_path)
    if not source_path.is_file():
        return
    with h5py.File(source_path, 'r') as source:
        _copy_attrs(source, destination)

        def copy_item(name, item):
            leaf = name.rsplit('/', 1)[-1]
            if isinstance(item, h5py.Group):
                group = destination.require_group(name)
                _copy_attrs(item, group)
                return
            if leaf in ('pixels', 'pixel'):
                return
            parent_name, _, dataset_name = name.rpartition('/')
            parent = destination.require_group(parent_name) if parent_name else destination
            chunks = item.chunks if item.ndim and all(size > 0 for size in item.shape) else None
            copied = parent.create_dataset(
                dataset_name,
                shape=item.shape,
                dtype=item.dtype,
                chunks=chunks,
            )
            _copy_attrs(item, copied)
            if item.ndim == 0:
                copied[()] = item[()]
                return
            if not item.shape[0]:
                return
            bytes_per_row = max(1, int(np.prod(item.shape[1:] or (1,))) * item.dtype.itemsize)
            rows_per_copy = max(1, (64 * 1024 * 1024) // bytes_per_row)
            for start in range(0, item.shape[0], rows_per_copy):
                stop = min(start + rows_per_copy, item.shape[0])
                copied[start:stop] = item[start:stop]

        source.visititems(copy_item)


def _fixed_list_to_numpy(column):
    import pyarrow as pa

    if pa.types.is_fixed_size_list(column.type):
        return column.flatten().to_numpy(zero_copy_only=False).reshape(len(column), column.type.list_size)
    return np.asarray(column.to_pylist())


def read_lance_metadata(table):
    """Materialize the small, non-image columns required by latent training."""
    names = set(table.schema.names)
    if 'episode_idx' not in names:
        raise ValueError('Lance source must contain episode_idx.')
    columns = ['episode_idx']
    if 'step_idx' in names:
        columns.append('step_idx')
    if 'action' in names:
        columns.append('action')

    chunks = {name: [] for name in columns}
    for batch in table.to_lance().scanner(columns=columns).to_reader():
        for name in columns:
            column = batch.column(batch.schema.get_field_index(name))
            if name == 'action':
                values = _fixed_list_to_numpy(column)
            else:
                values = column.to_numpy(zero_copy_only=False)
            chunks[name].append(values)
    arrays = {name: np.concatenate(values) for name, values in chunks.items()}
    arrays['episode_idx'] = arrays['episode_idx'].astype(np.int64, copy=False)
    offsets, lengths = compute_episode_layout(arrays['episode_idx'])
    arrays['ep_offset'] = offsets
    arrays['ep_len'] = lengths

    expected_steps = np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
    if 'step_idx' in arrays:
        source_steps = arrays['step_idx'].astype(np.int64, copy=False)
        if not np.array_equal(source_steps, expected_steps):
            raise ValueError('Lance step_idx is not contiguous within each episode.')
        arrays['step_idx'] = source_steps
    else:
        arrays['step_idx'] = expected_steps
    if 'action' in arrays:
        arrays['action'] = np.asarray(arrays['action'])
    return arrays


class LancePixelReader:
    def __init__(self, lance_path, decode_workers):
        import lancedb
        from lancedb.permutation import Permutation

        self.path = Path(lance_path).expanduser().resolve()
        if self.path.suffix != '.lance' or not self.path.is_dir():
            raise ValueError(f'Expected a *.lance table directory, got {self.path}.')
        self.table = lancedb.connect(str(self.path.parent)).open_table(self.path.stem)
        if 'pixels' not in self.table.schema.names:
            raise ValueError(f'Lance source has no pixels column: {self.path}')
        self.row_count = int(self.table.count_rows())
        self.schema = str(self.table.schema)
        self._rows = Permutation.identity(self.table).select_columns(['pixels']).with_format('arrow')
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(decode_workers)))

    @staticmethod
    def _decode(blob):
        with Image.open(io.BytesIO(blob)) as image:
            return np.asarray(image.convert('RGB'), dtype=np.uint8)

    def fetch(self, start, stop):
        rows = list(range(int(start), int(stop)))
        batch = self._rows.__getitems__(rows)
        column = batch.column(batch.schema.get_field_index('pixels'))
        blobs = column.to_pylist()
        return np.stack(list(self._executor.map(self._decode, blobs)))

    def close(self):
        self._executor.shutdown(wait=True)


def _ensure_array(file, name, values):
    values = np.asarray(values)
    if name in file:
        current = file[name]
        if current.shape != values.shape or not np.array_equal(current[...], values):
            raise ValueError(f'Source HDF5 field {name!r} disagrees with Lance metadata.')
        return
    chunks = True if values.ndim and values.shape[0] else None
    file.create_dataset(name, data=values, chunks=chunks)


def initialize_partial_cache(
    partial_path,
    *,
    args,
    reader,
    lance_arrays,
    checkpoint_path,
    checkpoint_sha256,
    checkpoint_metadata,
):
    import h5py

    source_hdf5 = reader.path.with_suffix('.h5')
    output_dtype = np.dtype(args.output_dtype)
    embed_dim = int(checkpoint_metadata['config']['embed_dim'])
    with h5py.File(partial_path, 'w') as output:
        copy_non_pixel_hdf5(source_hdf5, output)
        output.attrs['format'] = 'lewm_latent_dataset'
        output.attrs['format_version'] = FORMAT_VERSION
        output.attrs['status'] = 'initializing'
        output.attrs['task'] = args.task
        output.attrs['source_lance'] = str(reader.path)
        output.attrs['source_lance_rows'] = reader.row_count
        output.attrs['source_lance_schema'] = reader.schema
        output.attrs['source_hdf5'] = str(source_hdf5) if source_hdf5.is_file() else ''
        output.attrs['checkpoint_path'] = str(checkpoint_path)
        output.attrs['checkpoint_sha256'] = checkpoint_sha256
        output.attrs['checkpoint_epoch'] = int(checkpoint_metadata['epoch'])
        output.attrs['checkpoint_config_json'] = json.dumps(checkpoint_metadata['config'], sort_keys=True, default=str)
        output.attrs['architecture'] = str(checkpoint_metadata['config']['architecture'])
        output.attrs['embed_dim'] = embed_dim
        output.attrs['image_size'] = int(checkpoint_metadata['config']['image_size'])
        output.attrs['history_size'] = int(checkpoint_metadata['config']['history_size'])
        output.attrs['z_dtype'] = output_dtype.name
        output.attrs['encoded_rows'] = 0
        output.attrs['created_at_utc'] = datetime.now(timezone.utc).isoformat()
        output.attrs['command'] = ' '.join(shlex.quote(value) for value in sys.argv)

        _ensure_array(output, 'episode_idx', lance_arrays['episode_idx'])
        _ensure_array(output, 'step_idx', lance_arrays['step_idx'])
        _ensure_array(output, 'ep_offset', lance_arrays['ep_offset'])
        _ensure_array(output, 'ep_len', lance_arrays['ep_len'])
        if 'action' not in output and 'action' in lance_arrays:
            output.create_dataset('action', data=lance_arrays['action'], chunks=True)
        _ensure_array(output, 'source_row', np.arange(reader.row_count, dtype=np.int64))

        z = output.create_dataset(
            'z',
            shape=(reader.row_count, embed_dim),
            dtype=output_dtype,
            chunks=(min(8192, reader.row_count), embed_dim),
        )
        z.attrs['definition'] = 'LeWM.encode_pixels(pixels, train=False)'
        z.attrs['checkpoint_sha256'] = checkpoint_sha256
        output.attrs['status'] = 'encoding'
        output.flush()


def validate_existing_cache(
    path,
    *,
    reader,
    checkpoint_sha256,
    embed_dim,
    output_dtype,
    require_complete,
):
    import h5py

    with h5py.File(path, 'r') as file:
        expected = {
            'format': 'lewm_latent_dataset',
            'format_version': FORMAT_VERSION,
            'source_lance': str(reader.path),
            'source_lance_rows': reader.row_count,
            'checkpoint_sha256': checkpoint_sha256,
            'embed_dim': embed_dim,
            'z_dtype': np.dtype(output_dtype).name,
        }
        for key, value in expected.items():
            if file.attrs.get(key) != value:
                raise ValueError(f'Existing cache metadata mismatch for {key}: {file.attrs.get(key)!r} != {value!r}')
        if 'z' not in file or file['z'].shape != (reader.row_count, embed_dim):
            raise ValueError(f'Existing cache has an invalid z dataset: {path}')
        status = str(file.attrs.get('status', ''))
        if require_complete and status != 'complete':
            raise ValueError(f'Existing final cache is not complete: {path}')
        if not require_complete and status not in ('encoding', 'complete'):
            raise ValueError(f'Partial cache cannot be resumed from status={status!r}: {path}')
        return status, int(file.attrs.get('encoded_rows', 0))


def encode_rows(reader, encode_pixels, *, start, stop, batch_size, on_batch):
    import jax
    import jax.numpy as jnp

    for batch_start in range(start, stop, batch_size):
        batch_stop = min(batch_start + batch_size, stop)
        pixels = reader.fetch(batch_start, batch_stop)
        latents = np.asarray(jax.device_get(encode_pixels(jnp.asarray(pixels))), dtype=np.float32)
        if latents.ndim != 2 or len(latents) != len(pixels):
            raise ValueError(f'Encoder returned shape {latents.shape} for pixels {pixels.shape}.')
        if not np.isfinite(latents).all():
            raise FloatingPointError(f'Encoder produced non-finite latents for rows [{batch_start}, {batch_stop}).')
        on_batch(batch_start, batch_stop, pixels, latents)


def finalize_statistics(path, batch_rows=65536):
    import h5py

    with h5py.File(path, 'r+') as file:
        z = file['z']
        count = 0
        total = np.zeros(z.shape[1], dtype=np.float64)
        total_sq = np.zeros(z.shape[1], dtype=np.float64)
        minimum = np.full(z.shape[1], np.inf, dtype=np.float64)
        maximum = np.full(z.shape[1], -np.inf, dtype=np.float64)
        for start in range(0, len(z), batch_rows):
            values = z[start : start + batch_rows].astype(np.float64)
            if not np.isfinite(values).all():
                raise FloatingPointError(f'Cached z contains non-finite values near row {start}.')
            count += len(values)
            total += values.sum(axis=0)
            total_sq += np.square(values).sum(axis=0)
            minimum = np.minimum(minimum, values.min(axis=0))
            maximum = np.maximum(maximum, values.max(axis=0))
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 0.0)
        z.attrs['mean'] = mean.astype(np.float32)
        z.attrs['std'] = np.sqrt(variance).astype(np.float32)
        z.attrs['min'] = minimum.astype(np.float32)
        z.attrs['max'] = maximum.astype(np.float32)
        file.attrs['status'] = 'complete'
        file.attrs['encoded_rows'] = len(z)
        file.attrs['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        file.flush()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.decode_workers <= 0:
        raise ValueError('batch-size and decode-workers must be positive.')
    if args.flush_every_batches <= 0 or args.log_every_batches <= 0:
        raise ValueError('flush/log intervals must be positive.')
    if args.smoke_rows < 0:
        raise ValueError('smoke-rows must be non-negative.')

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'LeWM checkpoint not found: {checkpoint_path}')
    output_path = Path(args.output).expanduser().resolve()
    partial_path = output_path.with_name(output_path.name + '.incomplete')

    from lewm_jax.checkpoints import load_frozen_lewm

    print(f'Loading checkpoint: {checkpoint_path}', flush=True)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    model, variables, checkpoint_metadata = load_frozen_lewm(checkpoint_path)
    embed_dim = int(checkpoint_metadata['config']['embed_dim'])
    image_size = int(checkpoint_metadata['config']['image_size'])

    import jax

    encode_pixels = jax.jit(lambda pixels: model.apply(variables, pixels, train=False, method=model.encode_pixels))
    print(
        f'JAX backend={jax.default_backend()} devices={jax.devices()} '
        f'embed_dim={embed_dim} checkpoint_sha256={checkpoint_sha256}',
        flush=True,
    )

    reader = LancePixelReader(args.lance_path, args.decode_workers)
    try:
        if not reader.row_count:
            raise ValueError(f'Lance source is empty: {reader.path}')

        if args.smoke_rows:
            smoke_rows = min(args.smoke_rows, reader.row_count)
            observed_shapes = set()
            all_latents = []

            def collect_smoke(_, __, pixels, latents):
                observed_shapes.add(tuple(pixels.shape[1:]))
                all_latents.append(latents)

            encode_rows(
                reader,
                encode_pixels,
                start=0,
                stop=smoke_rows,
                batch_size=args.batch_size,
                on_batch=collect_smoke,
            )
            values = np.concatenate(all_latents)
            if observed_shapes != {(image_size, image_size, 3)}:
                raise ValueError(
                    f'Image shape mismatch: checkpoint expects {(image_size, image_size, 3)}, '
                    f'observed {sorted(observed_shapes)}.'
                )
            print(
                f'Smoke test passed: rows={smoke_rows} z_shape={values.shape} '
                f'z_dtype={values.dtype} z_mean={values.mean():.6f} '
                f'z_std={values.std():.6f}; no cache was created.',
                flush=True,
            )
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            status, encoded_rows = validate_existing_cache(
                output_path,
                reader=reader,
                checkpoint_sha256=checkpoint_sha256,
                embed_dim=embed_dim,
                output_dtype=args.output_dtype,
                require_complete=True,
            )
            print(
                f'Complete cache already exists; skipping: {output_path} (status={status}, rows={encoded_rows})',
                flush=True,
            )
            return

        if partial_path.exists():
            status, encoded_rows = validate_existing_cache(
                partial_path,
                reader=reader,
                checkpoint_sha256=checkpoint_sha256,
                embed_dim=embed_dim,
                output_dtype=args.output_dtype,
                require_complete=False,
            )
            if status == 'complete':
                os.replace(partial_path, output_path)
                print(f'Recovered completed cache: {output_path}', flush=True)
                return
            print(f'Resuming partial cache at row {encoded_rows}: {partial_path}', flush=True)
        else:
            print('Reading Lance trajectory metadata...', flush=True)
            lance_arrays = read_lance_metadata(reader.table)
            if len(lance_arrays['episode_idx']) != reader.row_count:
                raise ValueError(
                    f'Lance metadata has {len(lance_arrays["episode_idx"])} rows, but pixels have {reader.row_count}.'
                )
            initialize_partial_cache(
                partial_path,
                args=args,
                reader=reader,
                lance_arrays=lance_arrays,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_metadata=checkpoint_metadata,
            )
            encoded_rows = 0
            print(
                f'Initialized cache: rows={reader.row_count} episodes={len(lance_arrays["ep_len"])} '
                f'z_shape=({reader.row_count}, {embed_dim}) dtype={args.output_dtype}',
                flush=True,
            )

        import h5py

        started = time.monotonic()
        batch_counter = 0
        with h5py.File(partial_path, 'r+') as output:
            z = output['z']

            def write_batch(batch_start, batch_stop, pixels, latents):
                nonlocal batch_counter
                if pixels.shape[1:] != (image_size, image_size, 3):
                    raise ValueError(
                        f'Checkpoint expects images {(image_size, image_size, 3)}, '
                        f'but rows [{batch_start}, {batch_stop}) are {pixels.shape[1:]}.'
                    )
                z[batch_start:batch_stop] = latents.astype(args.output_dtype)
                output.attrs['encoded_rows'] = batch_stop
                batch_counter += 1
                if batch_counter % args.flush_every_batches == 0:
                    output.flush()
                if batch_counter % args.log_every_batches == 0 or batch_stop == reader.row_count:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    completed = batch_stop - encoded_rows
                    rate = completed / elapsed
                    remaining = (reader.row_count - batch_stop) / max(rate, 1e-6)
                    print(
                        f'Encoded {batch_stop}/{reader.row_count} '
                        f'({100.0 * batch_stop / reader.row_count:.2f}%) '
                        f'rate={rate:.1f} rows/s eta={remaining / 60.0:.1f} min',
                        flush=True,
                    )

            encode_rows(
                reader,
                encode_pixels,
                start=encoded_rows,
                stop=reader.row_count,
                batch_size=args.batch_size,
                on_batch=write_batch,
            )
            output.flush()

        print('Verifying all cached latents and computing statistics...', flush=True)
        finalize_statistics(partial_path)
        os.replace(partial_path, output_path)
        elapsed = time.monotonic() - started
        size_gib = output_path.stat().st_size / (1024**3)
        print(
            f'Completed latent cache: {output_path} rows={reader.row_count} '
            f'z_dim={embed_dim} size={size_gib:.2f} GiB elapsed={elapsed / 60.0:.1f} min',
            flush=True,
        )
    finally:
        reader.close()


if __name__ == '__main__':
    main()
