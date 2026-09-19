"""Canonical closed-loop CEM controller used by LeWM++."""

from __future__ import annotations

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
from latent_path_flow_runtime_lewm_control import SubgoalGenerator

from lewm_jax import load_frozen_lewm


def subgoal_planning_horizon(subgoal_steps, action_block):
    """Return the number of action blocks needed to reach a local target."""
    if subgoal_steps <= 0 or action_block <= 0:
        raise ValueError('Subgoal steps and action block must be positive.')
    if subgoal_steps % action_block:
        raise ValueError('Subgoal steps must be divisible by the action block.')
    return subgoal_steps // action_block


def reduce_rollout_costs(distances, mode):
    """Reduce per-checkpoint latent distances to one cost per candidate."""
    if mode == 'last':
        return distances[..., -1]
    if mode == 'moh':
        return jnp.min(distances, axis=-1)
    raise ValueError(f'Unsupported CEM cost mode: {mode!r}.')


def _set_cem_anchors(candidates, mean, policy_anchor, anchor_policy):
    """Keep the current CEM mean and, optionally, the original policy mode."""
    candidates = candidates.at[0].set(mean)
    if anchor_policy:
        candidates = candidates.at[1].set(policy_anchor)
    return candidates


class LeWMPPController:
    """LeWM CEM with a selectable subgoal generator and action-prior mode."""

    def __init__(
        self,
        checkpoint,
        scaler,
        *,
        seed,
        horizon,
        receding_horizon,
        action_block,
        num_samples,
        iterations,
        topk,
        var_scale,
        cost_mode='moh',
        action_prior=None,
        action_prior_mode='zero',
        paired_plan_keys=True,
        action_low=None,
        action_high=None,
        subgoal_generator_checkpoint=None,
        flow_sampling_steps=16,
    ):
        if horizon <= 0 or receding_horizon <= 0 or action_block <= 0:
            raise ValueError('CEM horizon, receding horizon, and action block must be positive.')
        if num_samples <= 1 or iterations <= 0:
            raise ValueError('CEM requires at least two samples and one iteration.')
        if not 1 < topk <= num_samples:
            raise ValueError('CEM topk must be in [2, num_samples].')
        if var_scale <= 0:
            raise ValueError('CEM variance scale must be positive.')
        if cost_mode not in ('last', 'moh'):
            raise ValueError(f'Unsupported CEM cost mode: {cost_mode!r}.')
        if action_prior_mode not in ('zero', 'policy_mode', 'policy_mode_anchor'):
            raise ValueError(f'Unsupported action-prior mode: {action_prior_mode!r}.')
        if (action_prior is None) != (action_prior_mode == 'zero'):
            raise ValueError('Action prior must be absent exactly when action_prior_mode=zero.')
        if (action_low is None) != (action_high is None):
            raise ValueError('Action low and high bounds must be provided together.')

        model, variables, metadata = load_frozen_lewm(checkpoint)
        self.model = model
        self.variables = variables
        self.lewm_checkpoint = metadata['path']
        self.lewm_config = metadata['config']
        self.scaler = scaler
        self.seed = int(seed)
        self.rng = jax.random.PRNGKey(seed)
        self.action_block = int(action_block)
        self.receding_horizon = int(receding_horizon)
        self.num_samples = int(num_samples)
        self.iterations = int(iterations)
        self.topk = int(topk)
        self.var_scale = float(var_scale)
        self.cost_mode = str(cost_mode)
        self.action_prior = action_prior
        self.action_prior_mode = str(action_prior_mode)
        self.paired_plan_keys = bool(paired_plan_keys)
        prior_lewm_checkpoint = None if action_prior is None else getattr(action_prior, 'lewm_checkpoint', None)
        if prior_lewm_checkpoint is not None and str(prior_lewm_checkpoint) != self.lewm_checkpoint:
            raise ValueError('Action prior and planner must use the same LeWM checkpoint.')

        self._encode_pixels = jax.jit(
            lambda pixels: self.model.apply(
                self.variables,
                pixels,
                train=False,
                method=self.model.encode_pixels,
            ).astype(jnp.float32)
        )
        self.subgoal_generator = None
        if subgoal_generator_checkpoint is not None:
            self.subgoal_generator = SubgoalGenerator(
                subgoal_generator_checkpoint,
                self.encode_pixels,
                seed=self.seed,
                action_block=self.action_block,
                lewm_checkpoint=self.lewm_checkpoint,
                flow_sampling_steps=flow_sampling_steps,
            )

        self.requested_horizon = int(horizon)
        self.horizon = (
            self.requested_horizon
            if self.subgoal_generator is None
            else subgoal_planning_horizon(self.subgoal_generator.waypoint_step, self.action_block)
        )
        if self.receding_horizon > self.horizon:
            raise ValueError('CEM receding horizon cannot exceed the planning horizon.')

        self.planner_action_low = None
        self.planner_action_high = None
        if action_low is not None:
            action_low = np.asarray(action_low, dtype=np.float32)
            action_high = np.asarray(action_high, dtype=np.float32)
            expected_shape = (int(self.scaler.action_dim),)
            if action_low.shape != expected_shape or action_high.shape != expected_shape:
                raise ValueError(
                    f'Action bounds must have shape {expected_shape}; got {action_low.shape} and {action_high.shape}.'
                )
            if np.any(action_low >= action_high):
                raise ValueError('Each action low bound must be smaller than its high bound.')
            self.planner_action_low = np.tile(self.scaler.transform(action_low), self.action_block).astype(np.float32)
            self.planner_action_high = np.tile(self.scaler.transform(action_high), self.action_block).astype(np.float32)

        self._plan_one = jax.jit(self._build_plan_one())

    def encode_pixels(self, pixels):
        return np.asarray(self._encode_pixels(jnp.asarray(pixels)))

    @property
    def subgoal_generator_checkpoint(self):
        return None if self.subgoal_generator is None else self.subgoal_generator.checkpoint

    @property
    def subgoal_generator_checkpoint_step(self):
        return None if self.subgoal_generator is None else self.subgoal_generator.checkpoint_step

    @property
    def subgoal_generator_config(self):
        return None if self.subgoal_generator is None else self.subgoal_generator.config

    @property
    def flow_sampling_steps(self):
        return None if self.subgoal_generator is None else self.subgoal_generator.flow_sampling_steps

    @property
    def subgoal_generator_generation_counts(self):
        return (
            np.zeros(0, dtype=np.int64) if self.subgoal_generator is None else self.subgoal_generator.generation_counts
        )

    def _build_plan_one(self):
        model = self.model
        variables = self.variables
        num_samples = self.num_samples
        iterations = self.iterations
        topk = self.topk
        var_scale = self.var_scale
        cost_mode = self.cost_mode
        use_local_target = self.subgoal_generator is not None
        anchor_policy_mode = self.action_prior_mode == 'policy_mode_anchor'
        planner_action_low = (
            None if self.planner_action_low is None else jnp.asarray(self.planner_action_low, dtype=jnp.float32)
        )
        planner_action_high = (
            None if self.planner_action_high is None else jnp.asarray(self.planner_action_high, dtype=jnp.float32)
        )

        def plan_one(key, pixels, goals, target_embedding, initial_mean):
            def optimizer_step(_, carry):
                key, mean, std = carry
                key, sample_key = jax.random.split(key)
                candidates = (
                    jax.random.normal(
                        sample_key,
                        (num_samples, *initial_mean.shape),
                        dtype=jnp.float32,
                    )
                    * std[None]
                    + mean[None]
                )
                candidates = _set_cem_anchors(candidates, mean, initial_mean, anchor_policy_mode)
                if planner_action_low is not None:
                    candidates = jnp.clip(
                        candidates,
                        planner_action_low[None, None],
                        planner_action_high[None, None],
                    )
                goal_embeddings, predictions = model.apply(
                    variables,
                    pixels[None, None],
                    goals[None, None],
                    candidates[None],
                    method=model._rollout_predictions,
                )
                target = target_embedding[None, None, None] if use_local_target else goal_embeddings[:, None, None]
                distances = jnp.sum((predictions - target) ** 2, axis=-1)[0]
                costs = reduce_rollout_costs(distances, cost_mode)
                _, elite_indices = jax.lax.top_k(-costs, topk)
                elites = candidates[elite_indices]
                return key, elites.mean(axis=0), elites.std(axis=0, ddof=1)

            _, mean, _ = jax.lax.fori_loop(
                0,
                iterations,
                optimizer_step,
                (
                    key,
                    initial_mean,
                    jnp.full_like(initial_mean, var_scale),
                ),
            )
            if planner_action_low is not None:
                mean = jnp.clip(mean, planner_action_low, planner_action_high)
            return mean

        return plan_one

    def reset(self, action_space, num_envs):
        action_dim = int(np.prod(action_space.shape))
        if action_dim != self.scaler.action_dim:
            raise ValueError(f'Environment action dim {action_dim} differs from dataset dim {self.scaler.action_dim}.')
        self.atomic_action_dim = action_dim
        self.block_action_dim = action_dim * self.action_block
        if self.action_prior is not None and int(self.action_prior.action_horizon) != self.action_block:
            raise ValueError('Action-prior horizon must equal the LeWM action block.')
        self.buffers = [deque() for _ in range(num_envs)]
        self.warm_starts = [None] * num_envs
        self.plan_counts = np.zeros(num_envs, dtype=np.int64)
        if self.subgoal_generator is not None:
            self.subgoal_generator.reset(num_envs)

    def _next_plan_keys(self, env_index):
        if self.paired_plan_keys:
            base_key = jax.random.fold_in(jax.random.PRNGKey(self.seed), int(env_index))
            base_key = jax.random.fold_in(base_key, int(self.plan_counts[env_index]))
            self.plan_counts[env_index] += 1
            return jax.random.fold_in(base_key, 1), base_key
        if self.action_prior is None:
            self.rng, plan_key = jax.random.split(self.rng)
            return None, plan_key
        self.rng, prior_key, plan_key = jax.random.split(self.rng, 3)
        return prior_key, plan_key

    def _initial_mean(self, env_index, pixels, goals, prior_key):
        mean = np.zeros((self.horizon, self.block_action_dim), dtype=np.float32)
        warm_start = self.warm_starts[env_index]
        if warm_start is not None:
            mean[: len(warm_start)] = warm_start
        if self.action_prior is not None:
            block = np.asarray(
                self.action_prior.sample_actions(
                    observations=np.asarray(pixels[-1:]),
                    goals=np.asarray(goals[-1:]),
                    seed=prior_key,
                    temperature=0.0,
                )
            )
            if block.shape != (1, self.block_action_dim):
                raise ValueError(f'Action prior returned {block.shape}; expected (1, {self.block_action_dim}).')
            mean[0] = block[0]
        return mean

    def get_actions(self, pixels, goals, alive):
        for env_index in np.flatnonzero(alive):
            if self.subgoal_generator is not None:
                self.subgoal_generator.observe(env_index, np.asarray(pixels[env_index, -1]))
            if self.buffers[env_index]:
                continue

            prior_key, plan_key = self._next_plan_keys(env_index)
            target_embedding = np.zeros(int(self.lewm_config['embed_dim']), dtype=np.float32)
            if self.subgoal_generator is not None:
                target_embedding = self.subgoal_generator.predict_path(env_index, np.asarray(goals[env_index, -1]))[-1]
            initial_mean = self._initial_mean(env_index, pixels[env_index], goals[env_index], prior_key)
            normalized_blocks = np.asarray(
                self._plan_one(
                    plan_key,
                    jnp.asarray(pixels[env_index]),
                    jnp.asarray(goals[env_index]),
                    jnp.asarray(target_embedding),
                    jnp.asarray(initial_mean),
                )
            )
            keep = normalized_blocks[: self.receding_horizon]
            self.warm_starts[env_index] = normalized_blocks[self.receding_horizon :].copy()
            self.buffers[env_index].extend(self.scaler.inverse_transform(keep.reshape(-1, self.atomic_action_dim)))

        actions = np.full((len(alive), self.atomic_action_dim), np.nan, dtype=np.float32)
        for env_index in np.flatnonzero(alive):
            actions[env_index] = self.buffers[env_index].popleft()
        return actions
