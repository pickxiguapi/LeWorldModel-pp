"""Convert one visual stream from a LeRobotDataset v3 repository to LeWM Lance."""

from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def to_hwc_uint8(frame):
    """Convert a LeRobot image tensor/array to an RGB uint8 HWC array."""
    if hasattr(frame, 'detach'):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f'Expected a three-dimensional image, got {frame.shape}.')
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] not in (1, 3, 4):
        raise ValueError(f'Expected one, three, or four image channels, got {frame.shape}.')
    if np.issubdtype(frame.dtype, np.floating):
        finite = frame[np.isfinite(frame)]
        if not len(finite):
            raise ValueError('Image contains no finite pixels.')
        if float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0 + 1e-6:
            frame = frame * 255.0
        frame = np.rint(np.clip(frame, 0.0, 255.0)).astype(np.uint8)
    else:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def preprocess_frame(frame, image_size, resize_mode):
    image = Image.fromarray(to_hwc_uint8(frame))
    target = (int(image_size), int(image_size))
    if resize_mode == 'long_edge':
        scale = int(image_size) / max(image.size)
        target = tuple(max(1, round(dimension * scale)) for dimension in image.size)
        image = image.resize(target, resample=Image.Resampling.BILINEAR)
    elif resize_mode == 'center_crop':
        image = ImageOps.fit(image, target, method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))
    elif resize_mode == 'stretch':
        image = image.resize(target, resample=Image.Resampling.BILINEAR)
    else:
        raise ValueError(f'Unknown resize mode: {resize_mode!r}.')
    return np.asarray(image, dtype=np.uint8)


def encode_jpeg(frame, image_size, resize_mode, jpeg_quality):
    image = Image.fromarray(preprocess_frame(frame, image_size, resize_mode))
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=int(jpeg_quality))
    return output.getvalue()


def scalar_batch(values):
    if hasattr(values, 'detach'):
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1)


def vector_batch(values):
    if hasattr(values, 'detach'):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f'Expected a batch of action vectors, got {values.shape}.')
    return values


def fixed_list_array(values):
    import pyarrow as pa

    values = vector_batch(values)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


def lance_batch(batch, camera_key, image_size, resize_mode, jpeg_quality, encode_workers):
    import pyarrow as pa

    frames = batch[camera_key]
    if hasattr(frames, 'detach'):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)

    def encoder(frame):
        return encode_jpeg(frame, image_size, resize_mode, jpeg_quality)

    with ThreadPoolExecutor(max_workers=max(1, int(encode_workers))) as pool:
        pixels = list(pool.map(encoder, frames))
    episode_index = scalar_batch(batch['episode_index']).astype(np.int32, copy=False)
    frame_index = scalar_batch(batch['frame_index']).astype(np.int32, copy=False)
    source_index = scalar_batch(batch['index']).astype(np.int64, copy=False)
    actions = vector_batch(batch['action'])
    count = len(pixels)
    if not (len(episode_index) == len(frame_index) == len(source_index) == len(actions) == count):
        raise ValueError('LeRobot batch fields have different row counts.')
    return pa.Table.from_arrays(
        [
            pa.array(episode_index, type=pa.int32()),
            pa.array(frame_index, type=pa.int32()),
            pa.array(source_index, type=pa.int64()),
            pa.array(pixels, type=pa.binary()),
            fixed_list_array(actions),
        ],
        names=['episode_idx', 'step_idx', 'source_index', 'pixels', 'action'],
    )


def completed_rows(table):
    count = int(table.count_rows())
    if not count:
        return 0
    batch = table.to_lance().take([count - 1], columns=['source_index'])
    source_index = int(batch.column(0)[0].as_py())
    if source_index != count - 1:
        raise ValueError(
            f'Existing Lance source_index ends at {source_index}, expected {count - 1}; '
            'refusing an unsafe resume.'
        )
    return count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--camera-key', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument(
        '--resize-mode',
        choices=('long_edge', 'center_crop', 'stretch'),
        default='long_edge',
    )
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--decode-workers', type=int, default=4)
    parser.add_argument('--encode-workers', type=int, default=8)
    parser.add_argument('--video-backend', default='pyav')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.root:
        raise ValueError('--root must be filled in with a machine-local LeRobot dataset directory.')
    if args.image_size <= 0 or args.batch_size <= 0:
        raise ValueError('Image size and batch size must be positive.')
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError('JPEG quality must be in [1, 100].')

    destination = Path(args.destination).expanduser().resolve()
    if destination.suffix != '.lance':
        raise ValueError(f'Destination must end in .lance: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader, Subset

    dataset = LeRobotDataset(
        args.repo_id,
        root=Path(args.root).expanduser().resolve(),
        revision=args.revision,
        video_backend=args.video_backend,
    )
    if dataset.meta.info.get('codebase_version') != 'v3.0':
        raise ValueError(f'Expected LeRobotDataset v3.0, got {dataset.meta.info.get("codebase_version")!r}.')
    if args.camera_key not in dataset.meta.camera_keys:
        raise ValueError(f'Camera {args.camera_key!r} not found; choose one of {dataset.meta.camera_keys}.')
    action_feature = dataset.meta.features.get('action')
    if action_feature is None or len(action_feature.get('shape', ())) != 1:
        raise ValueError(f'Expected one vector action feature, got {action_feature!r}.')

    # LeRobot downloads every declared video stream to validate the local v3 snapshot.
    # Decode only the selected stream during conversion.
    for key in tuple(dataset.meta.video_keys):
        if key != args.camera_key:
            del dataset.meta.info['features'][key]

    import lancedb

    database = lancedb.connect(str(destination.parent))
    table = database.open_table(destination.stem) if destination.exists() else None
    completed = completed_rows(table) if table is not None else 0
    if completed > len(dataset):
        raise ValueError(f'Existing Lance has {completed} rows, but source has only {len(dataset)}.')
    print(
        f'repo={args.repo_id}@{args.revision} camera={args.camera_key} '
        f'frames={len(dataset)} episodes={dataset.meta.total_episodes} resume_row={completed}',
        flush=True,
    )

    remaining = Subset(dataset, range(completed, len(dataset)))
    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': args.decode_workers,
    }
    if args.decode_workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(remaining, **loader_kwargs)
    converted = completed
    for batch in loader:
        arrow_batch = lance_batch(
            batch,
            args.camera_key,
            args.image_size,
            args.resize_mode,
            args.jpeg_quality,
            args.encode_workers,
        )
        if table is None:
            table = database.create_table(destination.stem, data=arrow_batch)
        else:
            table.add(arrow_batch)
        converted += len(arrow_batch)
        print(f'{converted}/{len(dataset)} frames converted', flush=True)

    metadata = {
        'format': 'lewm_lerobot_v3_lance',
        'repo_id': args.repo_id,
        'revision': args.revision,
        'source_codebase_version': dataset.meta.info['codebase_version'],
        'robot_type': dataset.meta.robot_type,
        'fps': dataset.meta.fps,
        'total_episodes': dataset.meta.total_episodes,
        'total_frames': len(dataset),
        'camera_key': args.camera_key,
        'camera_source_shape': dataset.meta.features[args.camera_key]['shape'],
        'image_size': args.image_size,
        'resize_mode': args.resize_mode,
        'jpeg_quality': args.jpeg_quality,
        'action_shape': action_feature['shape'],
        'action_names': action_feature.get('names'),
        'destination': str(destination),
    }
    sidecar = destination.parent / f'{destination.stem}.conversion.json'
    sidecar.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Wrote {destination} and {sidecar}', flush=True)


if __name__ == '__main__':
    main()
