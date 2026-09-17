"""Precompute frozen LeWM latents from an official OGBench visual NPZ file."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from precompute_lewm_lance_latents import finalize_statistics, sha256_file

FORMAT_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-name', required=True)
    parser.add_argument('--npz-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=512)
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


def episode_layout_from_terminals(terminals):
    """Build contiguous trajectory metadata from terminal rows."""
    terminals = np.asarray(terminals, dtype=bool)
    if terminals.ndim != 1 or not len(terminals):
        raise ValueError('terminals must be a non-empty one-dimensional array.')
    ends = np.flatnonzero(terminals).astype(np.int64) + 1
    if not len(ends) or int(ends[-1]) != len(terminals):
        ends = np.concatenate((ends, np.asarray([len(terminals)], dtype=np.int64)))
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
    lengths = np.diff(np.concatenate((offsets, np.asarray([len(terminals)]))))
    if np.any(lengths <= 0):
        raise ValueError('Terminal rows produced an empty episode.')
    episode_idx = np.repeat(
        np.arange(len(lengths), dtype=np.int64), lengths
    )
    step_idx = np.concatenate(
        [np.arange(length, dtype=np.int64) for length in lengths]
    )
    return episode_idx, step_idx, offsets, lengths


def load_npz_source(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'OGBench NPZ dataset not found: {path}')
    archive = np.load(path, allow_pickle=False)
    required = {'observations', 'actions', 'terminals'}
    missing = required.difference(archive.files)
    if missing:
        archive.close()
        raise ValueError(f'{path} is missing arrays: {sorted(missing)}')
    arrays = {name: archive[name] for name in archive.files}
    observations = arrays['observations']
    terminals = np.asarray(arrays['terminals'], dtype=bool)
    if observations.ndim != 4 or observations.shape[-1] not in (1, 3, 4):
        archive.close()
        raise ValueError(f'Expected HWC observations, got {observations.shape}.')
    if observations.dtype != np.uint8:
        archive.close()
        raise ValueError(f'Expected uint8 observations, got {observations.dtype}.')
    if any(np.asarray(value).ndim == 0 or len(value) != len(observations) for value in arrays.values()):
        archive.close()
        raise ValueError('Every NPZ array must have one leading row per observation.')
    if terminals.shape != (len(observations),):
        archive.close()
        raise ValueError('terminals must have one scalar per observation.')
    layout = episode_layout_from_terminals(terminals)
    return archive, arrays, layout


def create_dataset(output, name, values):
    values = np.asarray(values)
    chunks = None
    if values.ndim and len(values):
        row_bytes = max(1, int(np.prod(values.shape[1:] or (1,))) * values.dtype.itemsize)
        rows = min(len(values), max(1, (8 * 1024 * 1024) // row_bytes))
        chunks = (rows, *values.shape[1:])
    output.create_dataset(name, data=values, chunks=chunks)


def initialize_partial_cache(
    path,
    *,
    args,
    source_path,
    source_sha256,
    arrays,
    layout,
    checkpoint_path,
    checkpoint_sha256,
    checkpoint_metadata,
):
    import h5py

    observations = arrays['observations']
    episode_idx, step_idx, offsets, lengths = layout
    embed_dim = int(checkpoint_metadata['config']['embed_dim'])
    output_dtype = np.dtype(args.output_dtype)
    with h5py.File(path, 'w') as output:
        for name, values in arrays.items():
            if name != 'observations':
                create_dataset(output, name, values)
        create_dataset(output, 'episode_idx', episode_idx)
        create_dataset(output, 'step_idx', step_idx)
        create_dataset(output, 'ep_offset', offsets)
        create_dataset(output, 'ep_len', lengths)
        create_dataset(output, 'source_row', np.arange(len(observations), dtype=np.int64))

        output.attrs['format'] = 'lewm_latent_dataset'
        output.attrs['format_version'] = FORMAT_VERSION
        output.attrs['status'] = 'encoding'
        output.attrs['task'] = args.env_name
        output.attrs['source_format'] = 'ogbench_npz'
        output.attrs['source_npz'] = str(source_path)
        output.attrs['source_npz_sha256'] = source_sha256
        output.attrs['source_npz_rows'] = len(observations)
        output.attrs['source_npz_keys_json'] = json.dumps(sorted(arrays))
        output.attrs['checkpoint_path'] = str(checkpoint_path)
        output.attrs['checkpoint_sha256'] = checkpoint_sha256
        output.attrs['checkpoint_epoch'] = int(checkpoint_metadata['epoch'])
        output.attrs['checkpoint_config_json'] = json.dumps(
            checkpoint_metadata['config'], sort_keys=True, default=str
        )
        output.attrs['architecture'] = str(checkpoint_metadata['config']['architecture'])
        output.attrs['embed_dim'] = embed_dim
        output.attrs['image_size'] = int(checkpoint_metadata['config']['image_size'])
        output.attrs['history_size'] = int(checkpoint_metadata['config']['history_size'])
        output.attrs['z_dtype'] = output_dtype.name
        output.attrs['encoded_rows'] = 0
        output.attrs['created_at_utc'] = datetime.now(timezone.utc).isoformat()
        output.attrs['command'] = ' '.join(shlex.quote(value) for value in sys.argv)

        z = output.create_dataset(
            'z',
            shape=(len(observations), embed_dim),
            dtype=output_dtype,
            chunks=(min(8192, len(observations)), embed_dim),
        )
        z.attrs['definition'] = 'LeWM.encode_pixels(observations, train=False)'
        z.attrs['checkpoint_sha256'] = checkpoint_sha256
        output.flush()


def validate_existing_cache(
    path,
    *,
    source_path,
    source_sha256,
    num_rows,
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
            'source_format': 'ogbench_npz',
            'source_npz': str(source_path),
            'source_npz_sha256': source_sha256,
            'source_npz_rows': num_rows,
            'checkpoint_sha256': checkpoint_sha256,
            'embed_dim': embed_dim,
            'z_dtype': np.dtype(output_dtype).name,
        }
        for key, value in expected.items():
            if file.attrs.get(key) != value:
                raise ValueError(
                    f'Existing cache metadata mismatch for {key}: '
                    f'{file.attrs.get(key)!r} != {value!r}'
                )
        if 'z' not in file or file['z'].shape != (num_rows, embed_dim):
            raise ValueError(f'Existing cache has an invalid z dataset: {path}')
        status = str(file.attrs.get('status', ''))
        if require_complete and status != 'complete':
            raise ValueError(f'Existing final cache is not complete: {path}')
        if not require_complete and status not in ('encoding', 'complete'):
            raise ValueError(f'Partial cache cannot resume from status={status!r}: {path}')
        return status, int(file.attrs.get('encoded_rows', 0))


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError('batch-size must be positive.')
    if args.flush_every_batches <= 0 or args.log_every_batches <= 0:
        raise ValueError('flush/log intervals must be positive.')
    if args.smoke_rows < 0:
        raise ValueError('smoke-rows must be non-negative.')

    source_path = Path(args.npz_path).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    partial_path = output_path.with_name(output_path.name + '.incomplete')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'LeWM checkpoint not found: {checkpoint_path}')

    print(f'Hashing source NPZ: {source_path}', flush=True)
    source_sha256 = sha256_file(source_path)
    print(f'Loading source NPZ: {source_path}', flush=True)
    archive, arrays, layout = load_npz_source(source_path)
    observations = arrays['observations']
    try:
        from lewm_jax.checkpoints import load_frozen_lewm

        print(f'Loading checkpoint: {checkpoint_path}', flush=True)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        model, variables, metadata = load_frozen_lewm(checkpoint_path)
        embed_dim = int(metadata['config']['embed_dim'])
        image_size = int(metadata['config']['image_size'])
        if observations.shape[1:] != (image_size, image_size, 3):
            raise ValueError(
                f'Checkpoint expects {(image_size, image_size, 3)}, '
                f'but observations are {observations.shape[1:]}.'
            )

        import jax
        import jax.numpy as jnp

        encode_pixels = jax.jit(
            lambda pixels: model.apply(
                variables, pixels, train=False, method=model.encode_pixels
            )
        )
        print(
            f'JAX backend={jax.default_backend()} devices={jax.devices()} '
            f'rows={len(observations)} episodes={len(layout[3])} '
            f'embed_dim={embed_dim} checkpoint_sha256={checkpoint_sha256}',
            flush=True,
        )

        if args.smoke_rows:
            count = min(args.smoke_rows, len(observations))
            latents = np.asarray(
                jax.device_get(encode_pixels(jnp.asarray(observations[:count]))),
                dtype=np.float32,
            )
            if latents.shape != (count, embed_dim) or not np.isfinite(latents).all():
                raise ValueError(f'Invalid smoke latent output: {latents.shape}.')
            print(
                f'Smoke test passed: rows={count} z_shape={latents.shape} '
                f'z_mean={latents.mean():.6f} z_std={latents.std():.6f}; '
                'no cache was created.',
                flush=True,
            )
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        validation_kwargs = {
            'source_path': source_path,
            'source_sha256': source_sha256,
            'num_rows': len(observations),
            'checkpoint_sha256': checkpoint_sha256,
            'embed_dim': embed_dim,
            'output_dtype': args.output_dtype,
        }
        if output_path.exists():
            status, encoded_rows = validate_existing_cache(
                output_path, require_complete=True, **validation_kwargs
            )
            print(
                f'Complete cache already exists; skipping: {output_path} '
                f'(status={status}, rows={encoded_rows})',
                flush=True,
            )
            return
        if partial_path.exists():
            status, encoded_rows = validate_existing_cache(
                partial_path, require_complete=False, **validation_kwargs
            )
            if status == 'complete':
                os.replace(partial_path, output_path)
                print(f'Recovered completed cache: {output_path}', flush=True)
                return
            print(f'Resuming partial cache at row {encoded_rows}', flush=True)
        else:
            initialize_partial_cache(
                partial_path,
                args=args,
                source_path=source_path,
                source_sha256=source_sha256,
                arrays=arrays,
                layout=layout,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_metadata=metadata,
            )
            encoded_rows = 0
            print(f'Initialized partial cache: {partial_path}', flush=True)

        import h5py

        started = time.monotonic()
        batch_counter = 0
        with h5py.File(partial_path, 'r+') as output:
            z = output['z']
            for start in range(encoded_rows, len(observations), args.batch_size):
                stop = min(start + args.batch_size, len(observations))
                latents = np.asarray(
                    jax.device_get(
                        encode_pixels(jnp.asarray(observations[start:stop]))
                    ),
                    dtype=np.float32,
                )
                if latents.shape != (stop - start, embed_dim):
                    raise ValueError(
                        f'Encoder returned {latents.shape} for rows [{start}, {stop}).'
                    )
                if not np.isfinite(latents).all():
                    raise FloatingPointError(
                        f'Encoder produced non-finite latents near row {start}.'
                    )
                z[start:stop] = latents.astype(args.output_dtype)
                output.attrs['encoded_rows'] = stop
                batch_counter += 1
                if batch_counter % args.flush_every_batches == 0:
                    output.flush()
                if batch_counter % args.log_every_batches == 0 or stop == len(observations):
                    elapsed = max(time.monotonic() - started, 1e-6)
                    completed = stop - encoded_rows
                    rate = completed / elapsed
                    remaining = (len(observations) - stop) / max(rate, 1e-6)
                    print(
                        f'Encoded {stop}/{len(observations)} '
                        f'({100.0 * stop / len(observations):.2f}%) '
                        f'rate={rate:.1f} rows/s eta={remaining / 60.0:.1f} min',
                        flush=True,
                    )
            output.flush()

        print('Verifying all cached latents and computing statistics...', flush=True)
        finalize_statistics(partial_path)
        os.replace(partial_path, output_path)
        print(
            f'Completed latent cache: {output_path} rows={len(observations)} '
            f'z_dim={embed_dim} size={output_path.stat().st_size / (1024**3):.2f} GiB '
            f'elapsed={(time.monotonic() - started) / 60.0:.1f} min',
            flush=True,
        )
    finally:
        archive.close()


if __name__ == '__main__':
    main()
