"""Convert a LeWM HDF5 dataset to OGBench's indexed JPEG-backed Lance table."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import h5py
import lancedb
import numpy as np
import pyarrow as pa
from PIL import Image

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


def encode_frame(frame, jpeg_quality):
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 1:
        frame = frame[..., 0]
    output = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(output, format='JPEG', quality=jpeg_quality)
    return output.getvalue()


def fixed_list_array(values):
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


def episode_table(source, episode_indices, jpeg_quality):
    offsets = source['ep_offset'][:]
    lengths = source['ep_len'][:]
    episode_values = []
    step_values = []
    pixel_values = []
    action_values = []
    for episode_idx in episode_indices:
        offset = int(offsets[episode_idx])
        length = int(lengths[episode_idx])
        end = offset + length
        pixels = source['pixels'][offset:end]
        actions = np.asarray(source['action'][offset:end])
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        episode_values.extend([episode_idx] * length)
        step_values.extend(range(length))
        pixel_values.extend(encode_frame(frame, jpeg_quality) for frame in pixels)
        action_values.append(actions)
    actions = np.concatenate(action_values).astype(np.float32, copy=False)
    return pa.Table.from_arrays(
        [
            pa.array(episode_values, type=pa.int32()),
            pa.array(step_values, type=pa.int32()),
            pa.array(pixel_values, type=pa.binary()),
            fixed_list_array(actions),
        ],
        names=['episode_idx', 'step_idx', 'pixels', 'action'],
    )


def completed_episodes(table):
    maximum = -1
    for batch in table.to_lance().scanner(columns=['episode_idx']).to_reader():
        values = batch.column(0).to_numpy(zero_copy_only=False)
        if len(values):
            maximum = max(maximum, int(values.max()))
    return maximum + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--chunk-episodes', type=int, default=100)
    args = parser.parse_args()

    source_path = Path(args.source).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()
    if destination.suffix != '.lance':
        raise ValueError(f'Destination must end in .lance: {destination}')
    if args.chunk_episodes <= 0:
        raise ValueError('--chunk-episodes must be positive.')
    destination.parent.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(destination.parent))
    table = None
    completed = 0
    if destination.exists():
        table = db.open_table(destination.stem)
        completed = completed_episodes(table)

    with h5py.File(source_path, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024) as source:
        required = {'ep_offset', 'ep_len', 'pixels', 'action'}
        missing = required.difference(source)
        if missing:
            raise ValueError(f'{source_path} is missing columns: {sorted(missing)}')
        total = len(source['ep_len'])
        print(f'Resuming at episode {completed}/{total}', flush=True)
        for begin in range(completed, total, args.chunk_episodes):
            end = min(begin + args.chunk_episodes, total)
            batch = episode_table(source, range(begin, end), args.jpeg_quality)
            if table is None:
                table = db.create_table(destination.stem, data=batch)
            else:
                table.add(batch)
            print(f'{end}/{total} episodes converted', flush=True)


if __name__ == '__main__':
    main()
