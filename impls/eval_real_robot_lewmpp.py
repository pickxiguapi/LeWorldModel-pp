"""Real-robot evaluation API for LeWM++."""

from __future__ import annotations

import json
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


class RealRobotLeWMPPPolicy(_RealRobotPolicyBase):
    """LeWM++ with Action Prior, LatentPathFlow subgoals, and MoH planning."""

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

        scaler, action_low, action_high = load_training_action_spec(
            lance_path,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        action_prior = load_action_prior(
            lance_path,
            action_prior_dir,
            action_prior_step,
            lewm_checkpoint=lewm_checkpoint,
            expected_representation_mode='all',
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
            action_prior=action_prior,
            action_prior_mode='policy_mode',
            paired_plan_keys=True,
            action_low=action_low,
            action_high=action_high,
            subgoal_generator_checkpoint=latent_path_flow_checkpoint,
            flow_sampling_steps=flow_sampling_steps,
        )
        self._validate_training_protocol(
            controller,
            action_prior_dir=action_prior_dir,
            latent_path_flow_checkpoint=latent_path_flow_checkpoint,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
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
        action_prior_dir,
        latent_path_flow_checkpoint,
        validation_fraction,
        split_seed,
    ):
        lewm = controller.lewm_config
        prior = json.loads((Path(action_prior_dir).expanduser() / 'flags.json').read_text())
        flow = controller.subgoal_generator_config
        if not lewm.get('episode_split'):
            raise ValueError('LeWM checkpoint was not trained with episode-level splitting.')
        if not np.isclose(float(lewm['train_fraction']), 1.0 - validation_fraction):
            raise ValueError('LeWM train fraction does not match the evaluation dataset split.')
        if int(lewm['split_seed']) != split_seed:
            raise ValueError('LeWM split seed does not match evaluation.')
        if not np.isclose(float(prior['validation_fraction']), validation_fraction):
            raise ValueError('Action Prior validation fraction does not match evaluation.')
        if int(prior['episode_split_seed']) != split_seed:
            raise ValueError('Action Prior split seed does not match evaluation.')
        if not np.isclose(float(flow['train_fraction']), 1.0 - validation_fraction):
            raise ValueError('LatentPathFlow train fraction does not match evaluation.')
        if int(flow['split_seed']) != split_seed:
            raise ValueError('LatentPathFlow split seed does not match evaluation.')
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
        if controller.subgoal_generator_checkpoint != expected_flow:
            raise ValueError('Loaded LatentPathFlow checkpoint path does not match the requested checkpoint.')

    def reset(self, goal_observation, history_observations=()):
        """Reset the episode and seed LatentPathFlow with earlier observations."""
        super().reset(goal_observation)
        for observation in history_observations:
            self.controller.subgoal_generator.observe(0, self.prepare_image(observation))

    def plan_action_chunk(self, observations, goal_observation):
        """Plan the next 10 native actions from observations ordered oldest to newest."""
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
        """Compile planning without sending an action to hardware."""
        self.plan_action_chunk([observation], goal_observation)
        self.reset(goal_observation)


__all__ = [
    'ActionScaler',
    'PUSH_MULTI_RED_CUBE_ACTION_NAMES',
    'RealRobotLeWMPPPolicy',
    'decode_image_bytes',
    'extract_camera_frame',
    'preprocess_robot_image',
]
