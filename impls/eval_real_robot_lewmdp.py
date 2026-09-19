"""Real-robot inference API for LeWM++ with a Diffusion Policy action prior."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from eval_real_robot_lewm import (
    PUSH_MULTI_RED_CUBE_ACTION_NAMES,
    ActionScaler,
    _RealRobotPolicyBase,
    decode_image_bytes,
    extract_camera_frame,
    load_training_action_spec,
    preprocess_robot_image,
)

CAMERA_KEY = 'observation.images.camera_h'


class DiffusionPolicyPrior:
    """Adapt a goal-conditioned LeRobot Diffusion Policy to LeWM CEM."""

    def __init__(self, checkpoint, scaler, *, device='cuda:0', input_color='rgb'):
        import torch
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        self.torch = torch
        self.device = torch.device(device)
        self.policy = DiffusionPolicy.from_pretrained(str(checkpoint)).to(self.device).eval()
        config = self.policy.config
        expected = {
            'goal_conditioning': True,
            'n_obs_steps': 3,
            'horizon': 10,
            'n_action_steps': 10,
            'noise_scheduler_type': 'DDIM',
            'num_inference_steps': 20,
        }
        for name, value in expected.items():
            if getattr(config, name) != value:
                raise ValueError(f'Diffusion Policy {name}={getattr(config, name)!r}; expected {value!r}.')
        if list(config.input_features) != [CAMERA_KEY]:
            raise ValueError(f'Diffusion Policy must use only {CAMERA_KEY}.')
        if tuple(config.output_features['action'].shape) != (int(scaler.action_dim),):
            raise ValueError(
                f'Diffusion Policy action shape is {tuple(config.output_features["action"].shape)}, '
                f'expected {(int(scaler.action_dim),)}.'
            )

        input_shape = tuple(int(value) for value in config.input_features[CAMERA_KEY].shape)
        if len(input_shape) != 3 or input_shape[0] != 3:
            raise ValueError(f'Diffusion Policy camera shape must be CHW RGB, got {input_shape}.')
        self.image_hw = input_shape[1:]
        self.preprocess, self.postprocess = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor': {'device': str(self.device)}},
        )
        self.checkpoint = str(checkpoint)
        self.scaler = scaler
        self.input_color = input_color
        self.action_horizon = 10
        self.action_dim = int(scaler.action_dim)
        # Unlike the native Action Chunk Prior, this policy does not share or
        # depend on a particular frozen LeWM encoder.
        self.lewm_checkpoint = None
        self.history = deque(maxlen=3)
        self.goal = None

    def _prepare_image(self, observation):
        pixels = preprocess_robot_image(
            observation,
            (*self.image_hw, 3),
            input_color=self.input_color,
        )
        chw = np.ascontiguousarray(pixels.transpose(2, 0, 1))
        return self.torch.from_numpy(chw).float().div_(255.0)

    def reset(self, goal_observation, history_observations=()):
        self.goal = self._prepare_image(goal_observation)
        self.history.clear()
        for observation in history_observations:
            self.observe(observation)

    def observe(self, observation):
        self.history.append(self._prepare_image(observation))

    @staticmethod
    def _seed_value(seed):
        words = np.asarray(seed, dtype=np.uint32).reshape(-1)
        value = 0
        for word in words:
            value = ((value << 32) ^ int(word)) & ((1 << 63) - 1)
        return value

    def sample_actions(self, observations, goals, seed, temperature=0.0):
        """Return one normalized 10-action block for LeWM's CEM mean."""
        del observations, goals, temperature
        if self.goal is None or not self.history:
            raise RuntimeError('Install a goal and at least one observation before sampling the Diffusion Policy.')
        history = list(self.history)
        history = [history[0]] * (3 - len(history)) + history
        images = self.torch.stack([*history, self.goal]).unsqueeze(0)
        cuda_devices = []
        if self.device.type == 'cuda':
            cuda_devices = [self.device.index if self.device.index is not None else self.torch.cuda.current_device()]
        with self.torch.random.fork_rng(devices=cuda_devices), self.torch.inference_mode():
            torch_seed = self._seed_value(seed)
            self.torch.manual_seed(torch_seed)
            if self.device.type == 'cuda':
                self.torch.cuda.manual_seed(torch_seed)
            batch = self.preprocess({CAMERA_KEY: images})
            visual = batch[CAMERA_KEY].unsqueeze(2)
            from lerobot.utils.constants import OBS_IMAGES

            actions = self.policy.diffusion.generate_actions({OBS_IMAGES: visual})
            actions = self.postprocess(actions)
        native = self.torch.as_tensor(actions).detach().cpu().numpy().astype(np.float32, copy=False)
        expected = (1, self.action_horizon, self.action_dim)
        if native.shape != expected or not np.isfinite(native).all():
            raise RuntimeError(f'Diffusion Policy returned {native.shape}, expected {expected} finite actions.')
        normalized = self.scaler.transform(native[0]).astype(np.float32, copy=False)
        return normalized.reshape(1, self.action_horizon * self.action_dim)


class RealRobotLeWMDPPolicy(_RealRobotPolicyBase):
    """LeWM++ with LatentPathFlow, MoH, and a Diffusion Policy prior."""

    def __init__(
        self,
        *,
        lance_path,
        lewm_checkpoint,
        diffusion_policy_checkpoint,
        latent_path_flow_checkpoint,
        diffusion_device='cuda:0',
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
        from lewm_jax import checkpoint_image_shape
        from lewm_jax.planner_lewm_control import LeWMPPController

        scaler, action_low, action_high = load_training_action_spec(
            lance_path,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        diffusion_prior = DiffusionPolicyPrior(
            diffusion_policy_checkpoint,
            scaler,
            device=diffusion_device,
            input_color=input_color,
        )
        controller = LeWMPPController(
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
            action_prior=diffusion_prior,
            action_prior_mode='policy_mode',
            paired_plan_keys=True,
            action_low=action_low,
            action_high=action_high,
            subgoal_generator_checkpoint=latent_path_flow_checkpoint,
            flow_sampling_steps=flow_sampling_steps,
        )
        self._validate_training_protocol(
            controller,
            latent_path_flow_checkpoint=latent_path_flow_checkpoint,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        self.diffusion_prior = diffusion_prior
        self.action_chunk_size = 10
        self._finish_initialization(
            controller,
            scaler,
            action_low,
            action_high,
            checkpoint_image_shape(controller.lewm_config),
            input_color,
        )

    @staticmethod
    def _validate_training_protocol(
        controller,
        *,
        latent_path_flow_checkpoint,
        validation_fraction,
        split_seed,
    ):
        lewm = controller.lewm_config
        flow = controller.subgoal_generator_config
        if not lewm.get('episode_split'):
            raise ValueError('LeWM checkpoint was not trained with episode-level splitting.')
        if not np.isclose(float(lewm['train_fraction']), 1.0 - validation_fraction):
            raise ValueError('LeWM train fraction does not match the evaluation dataset split.')
        if int(lewm['split_seed']) != split_seed:
            raise ValueError('LeWM split seed does not match evaluation.')
        if not np.isclose(float(flow['train_fraction']), 1.0 - validation_fraction):
            raise ValueError('LatentPathFlow train fraction does not match evaluation.')
        if int(flow['split_seed']) != split_seed:
            raise ValueError('LatentPathFlow split seed does not match evaluation.')
        frameskip = int(lewm['frameskip'])
        action_block = int(flow['action_block'])
        subgoal_steps = int(flow['subgoal_steps'])
        if (frameskip, action_block, subgoal_steps) != (10, 10, 20):
            raise ValueError(
                'Real-robot checkpoints must use frameskip=10, action_block=10, and subgoal_steps=20.'
            )
        expected_flow = str(Path(latent_path_flow_checkpoint).expanduser().resolve())
        if controller.subgoal_generator_checkpoint != expected_flow:
            raise ValueError('Loaded LatentPathFlow checkpoint path does not match the requested checkpoint.')

    def reset(self, goal_observation, history_observations=()):
        """Reset LeWM++, Diffusion Policy history, and the LatentPathFlow history."""
        history_observations = list(history_observations)
        super().reset(goal_observation)
        self.diffusion_prior.reset(goal_observation, history_observations)
        for observation in history_observations:
            self.controller.subgoal_generator.observe(0, self.prepare_image(observation))

    def act(self, observation):
        """Append the latest DP observation and return one bounded native action."""
        self.diffusion_prior.observe(observation)
        return super().act(observation)

    def plan_action_chunk(self, observations, goal_observation):
        """Plan ten actions from observations ordered oldest to newest."""
        observations = list(observations)
        if not observations:
            raise ValueError('At least one observation is required.')
        self.reset(goal_observation, history_observations=observations[:-1])
        actions = self._drain_action_buffer(self.act(observations[-1]))
        expected = (self.action_chunk_size, self.action_dim)
        if actions.shape != expected:
            raise ValueError(f'Planned action chunk has shape {actions.shape}, expected {expected}.')
        return actions

    def warmup(self, observation, goal_observation):
        """Compile LeWM planning and execute one throwaway DP forward pass."""
        self.plan_action_chunk([observation] * 3, goal_observation)
        self.reset(goal_observation)


__all__ = [
    'ActionScaler',
    'DiffusionPolicyPrior',
    'PUSH_MULTI_RED_CUBE_ACTION_NAMES',
    'RealRobotLeWMDPPolicy',
    'decode_image_bytes',
    'extract_camera_frame',
    'preprocess_robot_image',
]
