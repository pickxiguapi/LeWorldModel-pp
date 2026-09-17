"""Streaming LeWM++ policy API for real-robot RGB observations."""

from __future__ import annotations

import io
import json
from pathlib import Path
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
        self.action_dim = int(len(self.mean))

    def transform(self, value):
        return (np.asarray(value) - self.mean) / self.scale

    def inverse_transform(self, value):
        return np.asarray(value) * self.scale + self.mean


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


def preprocess_robot_image(frame, image_shape, input_color='rgb'):
    """Apply the training-time aspect-preserving long-edge resize."""
    expected_height, expected_width, expected_channels = tuple(int(value) for value in image_shape)
    if expected_channels != 3:
        raise ValueError(f'Checkpoint expects unsupported image shape {image_shape}.')
    image = Image.fromarray(_to_rgb_uint8(frame, input_color))
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


class RealRobotLeWMPPPolicy:
    """Stateful real-robot policy returning one native 14-D action per call."""

    def __init__(
        self,
        *,
        lance_path,
        lewm_checkpoint,
        action_prior_dir,
        action_prior_step,
        latent_path_flow_checkpoint,
        seed=0,
        input_color='rgb',
        validation_fraction=0.04,
        split_seed=0,
        cem_num_samples=300,
        cem_iterations=5,
        cem_topk=30,
        cem_var_scale=1.0,
        flow_sampling_steps=16,
    ):
        from action_prior_runtime_lewm_control import load_action_prior
        from lewm_jax import checkpoint_image_shape
        from lewm_jax.planner_lewm_control import LeWMPPController
        from utils.lewm_dataset import LeWMLanceDataset

        self.input_color = input_color
        training_data = LeWMLanceDataset(
            lance_path,
            split='train',
            validation_fraction=validation_fraction,
            episode_split_seed=split_seed,
        )
        scaler = ActionScaler(training_data.action_mean, training_data.action_std)
        if scaler.action_dim != 14:
            raise ValueError(f'push_multi_red_cube requires 14-D actions, got {scaler.action_dim}.')
        raw_training_actions = scaler.inverse_transform(training_data.actions)
        action_low = raw_training_actions.min(axis=0)
        action_high = raw_training_actions.max(axis=0)
        constant = action_low >= action_high
        action_low = np.where(constant, action_low - 1e-6, action_low)
        action_high = np.where(constant, action_high + 1e-6, action_high)

        action_prior = load_action_prior(
            lance_path,
            action_prior_dir,
            action_prior_step,
            lewm_checkpoint=lewm_checkpoint,
            expected_representation_mode='all',
        )
        self.controller = LeWMPPController(
            checkpoint=lewm_checkpoint,
            scaler=scaler,
            seed=seed,
            horizon=2,
            receding_horizon=1,
            action_block=10,
            num_samples=cem_num_samples,
            iterations=cem_iterations,
            topk=cem_topk,
            var_scale=cem_var_scale,
            cost_mode='moh',
            action_prior=action_prior,
            action_prior_mode='policy_mode',
            paired_plan_keys=True,
            action_low=action_low,
            action_high=action_high,
            subgoal_generator_checkpoint=latent_path_flow_checkpoint,
            flow_sampling_steps=flow_sampling_steps,
        )
        self.image_shape = checkpoint_image_shape(self.controller.lewm_config)
        self._validate_training_protocol(
            action_prior_dir=action_prior_dir,
            latent_path_flow_checkpoint=latent_path_flow_checkpoint,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        self.action_dim = scaler.action_dim
        self.action_names = PUSH_MULTI_RED_CUBE_ACTION_NAMES
        self.action_low = action_low.astype(np.float32)
        self.action_high = action_high.astype(np.float32)
        self.goal_pixels = None
        self.controller.reset(SimpleNamespace(shape=(self.action_dim,)), 1)

    def _validate_training_protocol(
        self,
        *,
        action_prior_dir,
        latent_path_flow_checkpoint,
        validation_fraction,
        split_seed,
    ):
        lewm = self.controller.lewm_config
        prior = json.loads((Path(action_prior_dir).expanduser() / 'flags.json').read_text())
        flow = self.controller.subgoal_generator_config
        if not lewm.get('episode_split'):
            raise ValueError('LeWM checkpoint was not trained with episode-level splitting.')
        if not np.isclose(float(lewm['train_fraction']), 1.0 - validation_fraction):
            raise ValueError('LeWM train fraction does not match the deployment dataset split.')
        if int(lewm['split_seed']) != split_seed:
            raise ValueError('LeWM split seed does not match deployment.')
        if not np.isclose(float(prior['validation_fraction']), validation_fraction):
            raise ValueError('Action Prior validation fraction does not match deployment.')
        if int(prior['episode_split_seed']) != split_seed:
            raise ValueError('Action Prior split seed does not match deployment.')
        if not np.isclose(float(flow['train_fraction']), 1.0 - validation_fraction):
            raise ValueError('LatentPathFlow train fraction does not match deployment.')
        if int(flow['split_seed']) != split_seed:
            raise ValueError('LatentPathFlow split seed does not match deployment.')
        frameskip = int(lewm['frameskip'])
        chunk_size = int(prior['agent']['chunk_size'])
        action_block = int(flow['action_block'])
        subgoal_steps = int(flow['subgoal_steps'])
        if (frameskip, chunk_size, action_block, subgoal_steps) != (10, 10, 10, 20):
            raise ValueError(
                'Real-robot checkpoints must use frameskip=10, chunk_size=10, '
                'action_block=10, and subgoal_steps=20.'
            )
        expected_flow = str(Path(latent_path_flow_checkpoint).expanduser().resolve())
        if self.controller.subgoal_generator_checkpoint != expected_flow:
            raise ValueError('Loaded LatentPathFlow checkpoint path does not match the requested checkpoint.')

    def prepare_image(self, frame):
        return preprocess_robot_image(frame, self.image_shape, self.input_color)

    def reset(self, goal_image, history_frames=()):
        """Start an episode and optionally seed the latent history with earlier camera frames."""
        self.goal_pixels = self.prepare_image(goal_image)
        self.controller.reset(SimpleNamespace(shape=(self.action_dim,)), 1)
        for frame in history_frames:
            self.controller.subgoal_generator.observe(0, self.prepare_image(frame))

    def act(self, camera_h_frame):
        """Consume the latest RGB frame and return one bounded native robot action."""
        if self.goal_pixels is None:
            raise RuntimeError('Call reset(goal_image) before act(frame).')
        pixels = self.prepare_image(camera_h_frame)
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

    def warmup(self, camera_h_frame, goal_image):
        """Compile the complete planning path without sending an action to hardware."""
        self.plan_action_chunk([camera_h_frame], goal_image)
        self.reset(goal_image)

    def plan_action_chunk(self, camera_h_frames, goal_image):
        """Plan one action chunk from recent frames and a desired-goal image.

        Frames must be ordered from oldest to newest. The returned array has
        shape ``(10, 14)`` and uses ``self.action_names`` as its column order.
        """
        frames = list(camera_h_frames)
        if not frames:
            raise ValueError('At least one camera frame is required.')
        self.reset(goal_image, history_frames=frames[:-1])
        actions = [self.act(frames[-1])]
        while self.controller.buffers[0]:
            actions.append(np.asarray(self.controller.buffers[0].popleft(), dtype=np.float32))
        result = np.stack(actions)
        expected = (10, self.action_dim)
        if result.shape != expected:
            raise ValueError(f'Planned action chunk has shape {result.shape}, expected {expected}.')
        return np.clip(result, self.action_low, self.action_high)


def decode_image_bytes(value):
    """Convenience helper for adapters that receive encoded camera bytes."""
    with Image.open(io.BytesIO(value)) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8)
