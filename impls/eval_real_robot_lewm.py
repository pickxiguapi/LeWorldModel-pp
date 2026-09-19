"""Real-robot evaluation API for the LeWM baseline."""

from __future__ import annotations

import io
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
from PIL import Image

PUSH_MULTI_RED_CUBE_ACTION_NAMES = (
    'left_joint_0',
    'left_joint_1',
    'left_joint_2',
    'left_joint_3',
    'left_joint_4',
    'left_joint_5',
    'left_gripper',
    'right_joint_0',
    'right_joint_1',
    'right_joint_2',
    'right_joint_3',
    'right_joint_4',
    'right_joint_5',
    'right_gripper',
)


class ActionScaler:
    """Frozen action normalization fitted by the training-data loader."""

    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        if self.mean.ndim != 1 or self.scale.shape != self.mean.shape:
            raise ValueError('Action mean and scale must be one-dimensional arrays with identical shapes.')
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError('Action normalization statistics must be finite with strictly positive scales.')
        self.action_dim = int(len(self.mean))

    def transform(self, value):
        return (np.asarray(value) - self.mean) / self.scale

    def inverse_transform(self, value):
        return np.asarray(value) * self.scale + self.mean


def extract_camera_frame(observation):
    """Extract ``camera_h`` from a raw frame or a common robot-observation mapping."""
    if not isinstance(observation, Mapping):
        return observation
    for key in ('observation.images.camera_h', 'camera_h'):
        if key in observation:
            return observation[key]
    nested = observation.get('observation')
    if isinstance(nested, Mapping):
        if 'camera_h' in nested:
            return nested['camera_h']
        images = nested.get('images')
        if isinstance(images, Mapping) and 'camera_h' in images:
            return images['camera_h']
    raise KeyError(
        'Observation mapping must contain camera_h, observation.images.camera_h, '
        'or observation["images"]["camera_h"].'
    )


def _to_rgb_uint8(frame, input_color):
    if hasattr(frame, 'detach'):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f'Expected one HWC or CHW image, got {frame.shape}.')
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] not in (1, 3, 4):
        raise ValueError(f'Expected one, three, or four channels, got {frame.shape}.')
    if np.issubdtype(frame.dtype, np.floating):
        finite = frame[np.isfinite(frame)]
        if not len(finite):
            raise ValueError('Image contains no finite pixels.')
        if float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0 + 1e-6:
            frame = frame * 255.0
    frame = np.rint(np.clip(frame, 0.0, 255.0)).astype(np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]
    if input_color == 'bgr':
        frame = frame[..., ::-1]
    elif input_color != 'rgb':
        raise ValueError(f'input_color must be rgb or bgr, got {input_color!r}.')
    return frame


def preprocess_robot_image(observation, image_shape, input_color='rgb'):
    """Extract the camera frame and apply the training-time long-edge resize."""
    expected_height, expected_width, expected_channels = tuple(int(value) for value in image_shape)
    if expected_channels != 3:
        raise ValueError(f'Checkpoint expects unsupported image shape {image_shape}.')
    image = Image.fromarray(_to_rgb_uint8(extract_camera_frame(observation), input_color))
    long_edge = max(expected_height, expected_width)
    scale = long_edge / max(image.size)
    target = tuple(max(1, round(dimension * scale)) for dimension in image.size)
    image = image.resize(target, resample=Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.shape != (expected_height, expected_width, expected_channels):
        raise ValueError(
            f'Aspect ratio does not match training data: resized frame is {pixels.shape}, '
            f'checkpoint expects {(expected_height, expected_width, expected_channels)}.'
        )
    return pixels


def load_training_action_spec(lance_path, validation_fraction=0.04, split_seed=0):
    """Restore action normalization and native bounds from the training split."""
    from utils.lewm_dataset import LeWMLanceDataset

    training_data = LeWMLanceDataset(
        lance_path,
        split='train',
        validation_fraction=validation_fraction,
        episode_split_seed=split_seed,
    )
    scaler = ActionScaler(training_data.action_mean, training_data.action_std)
    raw_training_actions = scaler.inverse_transform(training_data.actions)
    action_low = raw_training_actions.min(axis=0)
    action_high = raw_training_actions.max(axis=0)
    constant = action_low >= action_high
    action_low = np.where(constant, action_low - 1e-6, action_low).astype(np.float32)
    action_high = np.where(constant, action_high + 1e-6, action_high).astype(np.float32)
    return scaler, action_low, action_high


class _RealRobotPolicyBase:
    def _finish_initialization(self, controller, scaler, action_low, action_high, image_shape, input_color):
        self.controller = controller
        self.image_shape = tuple(int(value) for value in image_shape)
        self.input_color = input_color
        self.action_dim = int(scaler.action_dim)
        if self.action_dim != len(PUSH_MULTI_RED_CUBE_ACTION_NAMES):
            raise ValueError(f'push_multi_red_cube requires 14-D actions, got {self.action_dim}.')
        self.action_names = PUSH_MULTI_RED_CUBE_ACTION_NAMES
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.goal_pixels = None
        self.controller.reset(SimpleNamespace(shape=(self.action_dim,)), 1)

    def prepare_image(self, observation):
        return preprocess_robot_image(observation, self.image_shape, self.input_color)

    def reset(self, goal_observation):
        """Reset controller state and install the desired-goal camera image."""
        self.goal_pixels = self.prepare_image(goal_observation)
        self.controller.reset(SimpleNamespace(shape=(self.action_dim,)), 1)

    def act(self, observation):
        """Consume one observation and return one bounded native 14-D action."""
        if self.goal_pixels is None:
            raise RuntimeError('Call reset(goal_observation) before act(observation).')
        pixels = self.prepare_image(observation)
        action = np.asarray(
            self.controller.get_actions(
                pixels[None, None],
                self.goal_pixels[None, None],
                np.asarray([True]),
            )[0],
            dtype=np.float32,
        )
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise FloatingPointError(f'Policy produced invalid action {action}.')
        return np.clip(action, self.action_low, self.action_high)

    def _drain_action_buffer(self, first_action):
        actions = [first_action]
        while self.controller.buffers[0]:
            actions.append(np.asarray(self.controller.buffers[0].popleft(), dtype=np.float32))
        return np.clip(np.stack(actions), self.action_low, self.action_high)


class RealRobotLeWMPolicy(_RealRobotPolicyBase):
    """LeWM baseline using final-goal CEM without Action Prior or subgoals."""

    def __init__(
        self,
        *,
        lance_path,
        lewm_checkpoint,
        seed=0,
        input_color='rgb',
        validation_fraction=0.04,
        split_seed=0,
        cem_horizon=5,
        cem_receding_horizon=5,
        action_block=10,
        cem_num_samples=300,
        cem_iterations=30,
        cem_topk=30,
        cem_var_scale=1.0,
    ):
        from lewm_jax import checkpoint_image_shape
        from lewm_jax.planner_lewm_control import LeWMPPController

        scaler, action_low, action_high = load_training_action_spec(
            lance_path,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        controller = LeWMPPController(
            checkpoint=lewm_checkpoint,
            scaler=scaler,
            seed=seed,
            horizon=cem_horizon,
            receding_horizon=cem_receding_horizon,
            action_block=action_block,
            num_samples=cem_num_samples,
            iterations=cem_iterations,
            topk=cem_topk,
            var_scale=cem_var_scale,
            cost_mode='last',
            action_prior=None,
            action_prior_mode='zero',
            paired_plan_keys=True,
            action_low=action_low,
            action_high=action_high,
            subgoal_generator_checkpoint=None,
        )
        self._validate_training_protocol(
            controller.lewm_config,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
            action_block=action_block,
        )
        self.action_chunk_size = int(cem_receding_horizon) * int(action_block)
        self._finish_initialization(
            controller,
            scaler,
            action_low,
            action_high,
            checkpoint_image_shape(controller.lewm_config),
            input_color,
        )

    @staticmethod
    def _validate_training_protocol(lewm_config, *, validation_fraction, split_seed, action_block):
        if not lewm_config.get('episode_split'):
            raise ValueError('LeWM checkpoint was not trained with episode-level splitting.')
        if not np.isclose(float(lewm_config['train_fraction']), 1.0 - validation_fraction):
            raise ValueError('LeWM train fraction does not match the evaluation dataset split.')
        if int(lewm_config['split_seed']) != split_seed:
            raise ValueError('LeWM split seed does not match evaluation.')
        if int(lewm_config['frameskip']) != action_block:
            raise ValueError('LeWM frameskip must equal the CEM action block.')

    def plan_action_chunk(self, observation, goal_observation):
        """Plan the next ``cem_receding_horizon * action_block`` native actions."""
        self.reset(goal_observation)
        actions = self._drain_action_buffer(self.act(observation))
        expected = (self.action_chunk_size, self.action_dim)
        if actions.shape != expected:
            raise ValueError(f'Planned action chunk has shape {actions.shape}, expected {expected}.')
        return actions

    def warmup(self, observation, goal_observation):
        """Compile planning without sending an action to hardware."""
        self.plan_action_chunk(observation, goal_observation)
        self.reset(goal_observation)


def decode_image_bytes(value):
    """Decode an encoded camera image into an RGB uint8 array."""
    with Image.open(io.BytesIO(value)) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8)


__all__ = [
    'ActionScaler',
    'PUSH_MULTI_RED_CUBE_ACTION_NAMES',
    'RealRobotLeWMPolicy',
    'decode_image_bytes',
    'extract_camera_frame',
    'preprocess_robot_image',
]
