import collections
import time
from typing import ClassVar

import gymnasium
import numpy as np
from gymnasium.spaces import Box

import ogbench
from utils.datasets import Dataset


class DatasetSpecEnv(gymnasium.Env):
    """Space-only environment used when training and evaluation are separate."""

    metadata: ClassVar[dict] = {}

    def __init__(self, observation_shape, observation_dtype, action_dim):
        self.observation_space = Box(
            low=0,
            high=255,
            shape=tuple(observation_shape),
            dtype=observation_dtype,
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(int(action_dim),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        return observation, {'goal': observation.copy()}

    def step(self, action):
        raise RuntimeError(
            'DatasetSpecEnv is training-only. Use the separate dataset-goal evaluator for Stable WM environments.'
        )


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(
    dataset_name,
    frame_stack=None,
    dataset_path=None,
    validation_fraction=0.05,
):
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the dataset.
        frame_stack: Number of frames to stack.

    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    if dataset_path is not None and dataset_path.endswith('.lance'):
        from utils.lewm_dataset import make_lewm_lance_datasets

        train_dataset, val_dataset = make_lewm_lance_datasets(dataset_path, validation_fraction=validation_fraction)
        env = DatasetSpecEnv(
            train_dataset.observations.shape[1:],
            train_dataset.observations.dtype,
            train_dataset.actions.shape[-1],
        )
    else:
        # Use compact dataset to save memory.
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            dataset_name, dataset_path=dataset_path, compact_dataset=True
        )
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env.reset()

    return env, train_dataset, val_dataset
