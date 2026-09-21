"""Dataset-goal evaluation runtime for OGBench's built-in LeWM environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import numpy as np
from PIL import Image

from ogbench.lewm_envs import ENV_IDS


@dataclass(frozen=True)
class TaskSpec:
    hdf5_name: str
    lance_name: str
    env_id: str
    env_kwargs: dict


TASK_SPECS = {
    'cube': TaskSpec(
        'cube_single_expert.h5',
        'cube_single_expert.lance',
        ENV_IDS['cube'],
        {
            'env_type': 'single',
            'ob_type': 'states',
            'multiview': False,
            'width': 224,
            'height': 224,
            'visualize_info': False,
            'terminate_at_goal': True,
        },
    ),
    'pusht': TaskSpec('pusht_expert_train.h5', 'pusht_expert_train.lance', ENV_IDS['pusht'], {}),
    'tworoom': TaskSpec('tworoom.h5', 'tworoom.lance', ENV_IDS['tworoom'], {}),
    'reacher': TaskSpec('reacher.h5', 'reacher.lance', ENV_IDS['reacher'], {'task': 'qpos_match'}),
}


class EvaluationPolicy(Protocol):
    def reset(self, action_space, num_envs: int) -> None: ...

    def get_actions(self, pixels: np.ndarray, goals: np.ndarray, alive: np.ndarray) -> np.ndarray: ...


class HDF5EvaluationDataset:
    """Small, lazy HDF5 reader exposing only the evaluation operations."""

    def __init__(self, path):
        import h5py

        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f'LeWM evaluation dataset not found: {self.path}')
        self._file = h5py.File(self.path, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024)
        self.column_names = []
        self._file.visititems(
            lambda name, value: self.column_names.append(name)
            if isinstance(value, h5py.Dataset) and name not in ('ep_len', 'ep_offset')
            else None
        )
        episode_col = 'episode_idx' if 'episode_idx' in self.column_names else 'ep_idx'
        if episode_col not in self.column_names or 'step_idx' not in self.column_names:
            raise ValueError(f'{self.path} must contain episode_idx/ep_idx and step_idx.')
        self.episode_column = episode_col
        if 'ep_offset' not in self._file or 'ep_len' not in self._file:
            raise ValueError(f'{self.path} must contain ep_offset and ep_len.')
        self.episode_offsets = np.asarray(self._file['ep_offset'][:], dtype=np.int64)
        self.episode_lengths = np.asarray(self._file['ep_len'][:], dtype=np.int64)
        if self.episode_offsets.shape != self.episode_lengths.shape:
            raise ValueError(f'{self.path} has inconsistent ep_offset and ep_len shapes.')
        self.episodes = np.asarray(self._file[episode_col][self.episode_offsets], dtype=np.int64)
        self._episode_slot = {int(episode): slot for slot, episode in enumerate(self.episodes)}
        if len(self._episode_slot) != len(self.episodes):
            raise ValueError(f'{self.path} contains duplicate episode identifiers.')

    def close(self):
        self._file.close()

    def get_column(self, name):
        return self._file[name][:]

    def row(self, episode, step):
        try:
            slot = self._episode_slot[int(episode)]
        except KeyError:
            raise KeyError(f'No row for episode={episode}, step={step} in {self.path}.') from None
        step = int(step)
        if step < 0 or step >= self.episode_lengths[slot]:
            raise KeyError(f'No row for episode={episode}, step={step} in {self.path}.')
        index = int(self.episode_offsets[slot] + step)
        if int(self._file['step_idx'][index]) != step:
            raise ValueError(f'{self.path} has non-contiguous step indices in episode {episode}.')
        return {name: self._file[name][index] for name in self.column_names}

    def sample_starts(self, num_eval, goal_offset, seed):
        if goal_offset < 0:
            raise ValueError(f'goal_offset must be non-negative, got {goal_offset}.')
        valid_counts = np.maximum(self.episode_lengths - goal_offset, 0)
        cumulative = np.cumsum(valid_counts)
        total_valid = int(cumulative[-1]) if len(cumulative) else 0
        if total_valid <= num_eval:
            raise ValueError(f'Only {total_valid} valid evaluation starts for {num_eval} runs.')
        rng = np.random.default_rng(seed)
        # Preserve the reference evaluator's exclusive upper bound.
        positions = np.sort(rng.choice(total_valid - 1, size=num_eval, replace=False))
        slots = np.searchsorted(cumulative, positions, side='right')
        previous = np.where(slots == 0, 0, cumulative[slots - 1])
        starts = positions - previous
        return self.episodes[slots].astype(int), starts.astype(int)


class StandardActionScaler:
    """scikit-learn StandardScaler semantics without a runtime dependency."""

    def __init__(self, actions):
        actions = np.asarray(actions)
        actions = actions[~np.isnan(actions).any(axis=1)]
        if not len(actions):
            raise ValueError('Cannot fit an action scaler without finite action rows.')
        self.mean = actions.mean(axis=0)
        self.scale = actions.std(axis=0, ddof=0)
        self.scale = np.where(self.scale > 0, self.scale, 1.0)
        self.action_dim = int(actions.shape[-1])

    def transform(self, value):
        return (np.asarray(value) - self.mean) / self.scale

    def inverse_transform(self, value):
        return np.asarray(value) * self.scale + self.mean


def _dataset_pixels(value):
    pixels = np.asarray(value)
    if pixels.ndim != 3:
        raise ValueError(f'Expected one HWC/CHW image, got shape {pixels.shape}.')
    if pixels.shape[0] in (1, 3) and pixels.shape[-1] not in (1, 3):
        pixels = np.moveaxis(pixels, 0, -1)
    return pixels.astype(np.uint8, copy=False)


def _resize_frame(value, image_size):
    frame = np.asarray(value)
    if frame.ndim != 3:
        raise ValueError(f'Environment render must be HWC, got shape {frame.shape}.')
    if frame.shape[:2] != (image_size, image_size):
        frame = np.asarray(Image.fromarray(frame).resize((image_size, image_size), Image.Resampling.BILINEAR))
    return frame.astype(np.uint8, copy=False)


def _set_dataset_state(task, env, init_row, goal_row):
    env = env.unwrapped
    if task == 'cube':
        env.set_state(qpos=np.asarray(init_row['qpos']), qvel=np.asarray(init_row['qvel']))
        env.set_target_pos(
            cube_id=0,
            target_pos=np.asarray(goal_row['privileged_block_0_pos']),
            target_quat=np.asarray(goal_row['privileged_block_0_quat']),
        )
    elif task == 'pusht':
        env._set_state(np.asarray(init_row['state']))
        env._set_goal_state(np.asarray(goal_row['state']))
    elif task == 'tworoom':
        env._set_state(np.asarray(init_row['proprio']))
        env._set_goal_state(np.asarray(goal_row['proprio']))
    elif task == 'reacher':
        env.set_state(np.asarray(init_row['qpos']), np.asarray(init_row['qvel']))
        env.set_target_qpos(np.asarray(goal_row['qpos']))
    else:
        raise ValueError(f'Unknown LeWM task: {task}')


def evaluate_dataset_goals(
    *,
    task,
    dataset,
    episodes,
    starts,
    goal_offset,
    eval_budget,
    policy,
    image_size=224,
    video_dir=None,
):
    """Evaluate one policy from fixed dataset states and future image goals."""
    spec = TASK_SPECS[task]
    envs = [
        gym.make(
            spec.env_id,
            max_episode_steps=2 * eval_budget,
            render_mode='rgb_array',
            **spec.env_kwargs,
        )
        for _ in episodes
    ]
    frames = [[] for _ in envs] if video_dir else None
    seeds = []
    goals = []
    try:
        for env, episode, start in zip(envs, episodes, starts):
            init_row = dataset.row(episode, start)
            goal_row = dataset.row(episode, start + goal_offset)
            seed = int(np.asarray(init_row['seed']).reshape(-1)[0]) if 'seed' in init_row else None
            seeds.append(seed)
            env.reset(seed=seed)
            _set_dataset_state(task, env, init_row, goal_row)
            goals.append(_resize_frame(_dataset_pixels(goal_row['pixels']), image_size))

        pixels = np.stack([_resize_frame(env.render(), image_size) for env in envs])
        goals = np.stack(goals)
        if frames is not None:
            for index, frame in enumerate(pixels):
                frames[index].append(frame.copy())

        alive = np.ones(len(envs), dtype=bool)
        successes = np.zeros(len(envs), dtype=bool)
        policy.reset(envs[0].action_space, len(envs))
        for _ in range(eval_budget):
            actions = np.asarray(policy.get_actions(pixels[:, None], goals[:, None], alive))
            expected = (len(envs), *envs[0].action_space.shape)
            if actions.shape != expected:
                raise ValueError(f'Policy action shape {actions.shape} does not match {expected}.')
            for index, env in enumerate(envs):
                if not alive[index]:
                    continue
                _, _, terminated, truncated, _ = env.step(actions[index])
                successes[index] |= bool(terminated)
                alive[index] &= not (terminated or truncated)
                pixels[index] = _resize_frame(env.render(), image_size)
                if frames is not None:
                    frames[index].append(pixels[index].copy())
            if not alive.any():
                break

        if frames is not None:
            import imageio.v3 as iio

            output = Path(video_dir)
            output.mkdir(parents=True, exist_ok=True)
            for index, episode_frames in enumerate(frames):
                iio.imwrite(
                    output / f'episode_{index}.mp4',
                    np.stack(episode_frames),
                    fps=10,
                )
        return {
            'success_rate': float(successes.mean() * 100.0),
            'episode_successes': successes,
            'seeds': seeds,
        }
    finally:
        for env in envs:
            env.close()


def task_paths(task, data_root):
    spec = TASK_SPECS[task]
    root = Path(data_root).expanduser().resolve()
    return root / spec.hdf5_name, root / spec.lance_name


def json_safe(value):
    """Convert NumPy containers to JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value
