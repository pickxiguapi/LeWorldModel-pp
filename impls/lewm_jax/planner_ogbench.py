"""LeWM-CEM controller used by the public evaluation entrypoints."""

from __future__ import annotations

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
from latent_path_flow_runtime_ogbench import LatentSubgoalGenerator

from lewm_jax import load_frozen_lewm


def subgoal_planning_horizon(subgoal_steps, action_block):
    """Return the number of action blocks needed to reach a subgoal."""
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
    if mode == 'path_mean':
        return jnp.mean(distances, axis=-1)
    raise ValueError(f'Unsupported CEM cost mode: {mode!r}.')


def policy_random_mixture_elites(costs, policy_count, topk, random_elite_cap):
    """Select CEM elites while capping random candidates.

    The first ``policy_count`` candidates are policy-guided.  We first retain
    ``topk - random_elite_cap`` of those candidates, then fill the remaining
    slots from the best still-unselected candidates.  This guarantees that no
    more than ``random_elite_cap`` random candidates can influence the refit.
    """
    policy_floor = topk - random_elite_cap
    _, policy_indices = jax.lax.top_k(-costs[:policy_count], policy_floor)
    remaining_costs = costs.at[policy_indices].set(jnp.inf)
    _, extra_indices = jax.lax.top_k(-remaining_costs, random_elite_cap)
    return jnp.concatenate((policy_indices, extra_indices), axis=0)


class JAXLeWMCEMPolicy:
    """Closed-loop LeWM-CEM with optional deterministic policy initialization."""

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
        cost_mode='last',
        guidance_policy=None,
        guidance_mode='none',
        guidance_population_size=0,
        guidance_temperature=1.0,
        guidance_first_block_std=None,
        guidance_random_elite_cap=0,
        guidance_mean_residual_weight=1.0,
        trace_candidates=False,
        collect_traces=True,
        action_low=None,
        action_high=None,
        latent_subgoal_checkpoint=None,
        latent_subgoal_flow_sampling_steps=None,
    ):
        if horizon <= 0 or receding_horizon <= 0 or action_block <= 0:
            raise ValueError(
                'CEM horizon, receding horizon, and action block must be positive.'
            )
        if num_samples <= 1 or iterations <= 0:
            raise ValueError('CEM requires at least two samples and one iteration.')
        if not 1 < topk <= num_samples:
            raise ValueError('CEM topk must be in [2, num_samples].')
        if var_scale <= 0:
            raise ValueError('CEM variance scale must be positive.')
        if cost_mode not in ('last', 'moh', 'path_mean'):
            raise ValueError(f'Unsupported CEM cost mode: {cost_mode!r}.')
        if guidance_mode not in ('none', 'policy_random_mixture'):
            raise ValueError(f'Unsupported policy guidance mode: {guidance_mode!r}.')
        if (guidance_policy is None) != (guidance_mode == 'none'):
            raise ValueError(
                'Policy guidance mode must be none exactly when no policy is provided.'
            )
        if not 0 <= int(guidance_population_size) <= int(num_samples):
            raise ValueError('Guidance population size must be in [0, CEM samples].')
        if guidance_mode == 'policy_random_mixture' and int(guidance_population_size) < 2:
            raise ValueError('Population-based guidance requires at least two proposals.')
        if guidance_mode == 'none' and int(guidance_population_size) != 0:
            raise ValueError(
                'Guidance population size only applies to population-based guidance.'
            )
        if float(guidance_temperature) < 0:
            raise ValueError('Guidance temperature must be non-negative.')
        if int(guidance_random_elite_cap) < 0:
            raise ValueError('Guidance random elite cap must be non-negative.')
        if guidance_mode == 'policy_random_mixture':
            random_count = int(num_samples) - int(guidance_population_size)
            if random_count <= 0:
                raise ValueError(
                    'Policy/random mixture requires at least one random candidate.'
                )
            if not 0 < int(guidance_random_elite_cap) < int(topk):
                raise ValueError(
                    'Policy/random mixture random elite cap must be in [1, topk).'
                )
            if int(guidance_random_elite_cap) > random_count:
                raise ValueError(
                    'Random elite cap cannot exceed the number of random candidates.'
                )
            if int(topk) - int(guidance_random_elite_cap) > int(
                guidance_population_size
            ):
                raise ValueError(
                    'Policy population is too small for the requested elite floor.'
                )
        elif int(guidance_random_elite_cap) != 0:
            raise ValueError(
                'Guidance random elite cap only applies to policy_random_mixture.'
            )
        if not 0.0 <= float(guidance_mean_residual_weight) <= 1.0:
            raise ValueError('Guidance mean residual weight must be in [0, 1].')
        if (
            guidance_mode != 'policy_random_mixture'
            and float(guidance_mean_residual_weight) != 1.0
        ):
            raise ValueError(
                'Guidance mean residual weight only applies to '
                'policy_random_mixture.'
            )
        if guidance_first_block_std is not None and float(guidance_first_block_std) <= 0:
            raise ValueError('Guidance first-block std must be positive.')
        if not hasattr(scaler, 'transform'):
            raise ValueError('Visual OGBench guidance requires a bidirectional action scaler.')
        if (action_low is None) != (action_high is None):
            raise ValueError('Action low and high bounds must be provided together.')

        model, variables, metadata = load_frozen_lewm(checkpoint)
        config = metadata['config']
        self.model = model
        self.variables = variables
        self.lewm_checkpoint = metadata['path']
        self.lewm_config = config
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
        self.guidance_policy = guidance_policy
        self.guidance_mode = str(guidance_mode)
        self.guidance_population_size = int(guidance_population_size)
        self.guidance_temperature = float(guidance_temperature)
        self.guidance_random_elite_cap = int(guidance_random_elite_cap)
        self.guidance_mean_residual_weight = float(
            guidance_mean_residual_weight
        )
        self.guidance_first_block_std = (
            None
            if guidance_first_block_std is None
            else float(guidance_first_block_std)
        )
        self.trace_candidates = bool(trace_candidates)
        self.collect_traces = bool(collect_traces)

        self._encode_pixels = jax.jit(
            lambda pixels: self.model.apply(
                self.variables,
                pixels,
                train=False,
                method=self.model.encode_pixels,
            ).astype(jnp.float32)
        )
        self.subgoal_generator = None
        if latent_subgoal_checkpoint is not None:
            self.subgoal_generator = LatentSubgoalGenerator(
                latent_subgoal_checkpoint,
                self.encode_pixels,
                seed=self.seed,
                action_block=self.action_block,
                lewm_checkpoint=self.lewm_checkpoint,
                flow_sampling_steps=latent_subgoal_flow_sampling_steps,
            )
        if self.cost_mode == 'path_mean' and self.subgoal_generator is None:
            raise ValueError('path_mean cost requires a latent subgoal path.')

        self.requested_horizon = int(horizon)
        if self.subgoal_generator is None:
            self.horizon = self.requested_horizon
        else:
            self.horizon = subgoal_planning_horizon(
                self.subgoal_generator.waypoint_step,
                self.action_block,
            )
        if (
            self.cost_mode == 'path_mean'
            and self.subgoal_generator.path_length != self.horizon
        ):
            raise ValueError(
                'path_mean requires one predicted waypoint per CEM action block: '
                f'{self.subgoal_generator.path_length} != {self.horizon}.'
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
                    f'Action bounds must have shape {expected_shape}; got '
                    f'{action_low.shape} and {action_high.shape}.'
                )
            if np.any(action_low >= action_high):
                raise ValueError(
                    'Each action low bound must be smaller than its high bound.'
                )
            self.planner_action_low = np.tile(
                self.scaler.transform(action_low), self.action_block
            ).astype(np.float32)
            self.planner_action_high = np.tile(
                self.scaler.transform(action_high), self.action_block
            ).astype(np.float32)

        self._plan_one = jax.jit(self._build_plan_one())

        model = self.model
        variables = self.variables

        def rollout_selected_plan(pixels, goals, plan):
            _, predictions = model.apply(
                variables,
                pixels[None, None],
                goals[None, None],
                plan[None, None],
                method=model._rollout_predictions,
            )
            return predictions[0, 0]

        self._rollout_selected_plan = jax.jit(rollout_selected_plan)

    def encode_pixels(self, pixels):
        return np.asarray(self._encode_pixels(jnp.asarray(pixels)))

    @property
    def latent_subgoal_checkpoint(self):
        return None if self.subgoal_generator is None else self.subgoal_generator.checkpoint

    @property
    def latent_subgoal_checkpoint_step(self):
        return (
            None
            if self.subgoal_generator is None
            else self.subgoal_generator.checkpoint_step
        )

    @property
    def latent_subgoal_config(self):
        return None if self.subgoal_generator is None else self.subgoal_generator.config

    @property
    def latent_subgoal_flow_sampling_steps(self):
        return (
            None
            if self.subgoal_generator is None
            else self.subgoal_generator.flow_sampling_steps
        )

    @property
    def latent_subgoal_waypoint_index(self):
        return None if self.subgoal_generator is None else self.subgoal_generator.waypoint_index

    @property
    def latent_subgoal_waypoint_step(self):
        return None if self.subgoal_generator is None else self.subgoal_generator.waypoint_step

    @property
    def latent_subgoal_history_size(self):
        return 0 if self.subgoal_generator is None else self.subgoal_generator.history_size

    @property
    def latent_subgoal_generation_counts(self):
        return (
            np.zeros(0, dtype=np.int64)
            if self.subgoal_generator is None
            else self.subgoal_generator.generation_counts
        )

    @property
    def latent_subgoal_trace(self):
        """Exact subgoal paths emitted at each closed-loop planning event."""
        return self.subgoal_traces

    def _build_plan_one(self):
        model = self.model
        variables = self.variables
        num_samples = self.num_samples
        iterations = self.iterations
        topk = self.topk
        var_scale = self.var_scale
        cost_mode = self.cost_mode
        use_subgoal = self.subgoal_generator is not None
        mixture_mode = self.guidance_mode == 'policy_random_mixture'
        guidance_population_size = self.guidance_population_size if mixture_mode else 0
        guidance_random_elite_cap = self.guidance_random_elite_cap
        guidance_mean_residual_weight = self.guidance_mean_residual_weight
        guidance_first_block_std = self.guidance_first_block_std
        planner_action_low = (
            None
            if self.planner_action_low is None
            else jnp.asarray(self.planner_action_low, dtype=jnp.float32)
        )
        planner_action_high = (
            None
            if self.planner_action_high is None
            else jnp.asarray(self.planner_action_high, dtype=jnp.float32)
        )
        trace_candidates = self.trace_candidates
        embed_dim = int(self.lewm_config['embed_dim'])
        horizon = self.horizon

        def plan_one(
            key,
            pixels,
            goals,
            target_embedding,
            initial_mean,
            guidance_blocks,
        ):
            initial_std = jnp.full_like(initial_mean, var_scale)
            if guidance_first_block_std is not None:
                initial_std = initial_std.at[0].set(guidance_first_block_std)

            def optimizer_step(iteration, carry):
                if trace_candidates:
                    key, mean, std, _, _, _ = carry
                elif mixture_mode:
                    key, mean, std, _ = carry
                else:
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
                candidates = candidates.at[0].set(mean)
                if mixture_mode:
                    candidates = candidates.at[
                        :guidance_population_size, 0
                    ].set(guidance_blocks[iteration])
                    # Candidate zero is the exact deterministic policy-mode plan;
                    # the remaining policy candidates keep stochastic CEM tails.
                    candidates = candidates.at[0].set(initial_mean)
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
                if use_subgoal:
                    target = (
                        target_embedding[None, None]
                        if cost_mode == 'path_mean'
                        else target_embedding[None, None, None]
                    )
                else:
                    target = goal_embeddings[:, None, None]
                distances = jnp.sum((predictions - target) ** 2, axis=-1)[0]
                costs = reduce_rollout_costs(distances, cost_mode)
                if mixture_mode:
                    elite_indices = policy_random_mixture_elites(
                        costs,
                        guidance_population_size,
                        topk,
                        guidance_random_elite_cap,
                    )
                else:
                    _, elite_indices = jax.lax.top_k(-costs, topk)
                elites = candidates[elite_indices]
                elite_mean = elites.mean(axis=0)
                if mixture_mode:
                    # Only the executable first block is anchored to the policy
                    # mode.  Future CEM blocks remain free to model the tail.
                    anchored_first = initial_mean[0] + (
                        guidance_mean_residual_weight
                        * (elite_mean[0] - initial_mean[0])
                    )
                    elite_mean = elite_mean.at[0].set(anchored_first)
                output = (key, elite_mean, elites.std(axis=0, ddof=1))
                if trace_candidates:
                    output += (
                        candidates,
                        predictions[0].astype(jnp.float32),
                        costs,
                    )
                elif mixture_mode:
                    output += (candidates[jnp.argmin(costs)],)
                return output

            if trace_candidates:
                initial_carry = (
                    key,
                    initial_mean,
                    initial_std,
                    jnp.zeros((num_samples, *initial_mean.shape), dtype=jnp.float32),
                    jnp.zeros(
                        (num_samples, horizon, embed_dim), dtype=jnp.float32
                    ),
                    jnp.zeros((num_samples,), dtype=jnp.float32),
                )
                _, mean, std, candidates, predictions, costs = jax.lax.fori_loop(
                    0, iterations, optimizer_step, initial_carry
                )
            elif mixture_mode:
                _, mean, std, best_candidate = jax.lax.fori_loop(
                    0,
                    iterations,
                    optimizer_step,
                    (
                        key,
                        initial_mean,
                        initial_std,
                        jnp.zeros_like(initial_mean),
                    ),
                )
                mean = best_candidate
            else:
                _, mean, std = jax.lax.fori_loop(
                    0,
                    iterations,
                    optimizer_step,
                    (key, initial_mean, initial_std),
                )
            if mixture_mode and trace_candidates:
                # Execute an actually evaluated final-pool trajectory rather than
                # the refitted mean, which may itself be off policy support.
                mean = candidates[jnp.argmin(costs)]
            if planner_action_low is not None:
                mean = jnp.clip(mean, planner_action_low, planner_action_high)
            if trace_candidates:
                return mean, std, candidates, predictions, costs
            return mean, std

        return plan_one

    def reset(self, action_space, num_envs):
        action_dim = int(np.prod(action_space.shape))
        if action_dim != self.scaler.action_dim:
            raise ValueError(
                f'Environment action dim {action_dim} differs from dataset dim '
                f'{self.scaler.action_dim}.'
            )
        self.atomic_action_dim = action_dim
        self.block_action_dim = action_dim * self.action_block
        if self.guidance_policy is not None:
            guidance_horizon = int(
                getattr(self.guidance_policy, 'action_horizon', 1)
            )
            if guidance_horizon != self.action_block:
                raise ValueError(
                    f'Guidance action horizon {guidance_horizon} differs from '
                    f'LeWM action block {self.action_block}.'
                )
        self.buffers = [deque() for _ in range(num_envs)]
        self.warm_starts = [None] * num_envs
        self.plan_counts = np.zeros(num_envs, dtype=np.int64)
        self.environment_steps = np.zeros(num_envs, dtype=np.int64)
        self.subgoal_traces = [[] for _ in range(num_envs)]
        if self.subgoal_generator is not None:
            self.subgoal_generator.reset(num_envs)

    def _next_plan_keys(self, env_index):
        plan_key = jax.random.fold_in(jax.random.PRNGKey(self.seed), int(env_index))
        plan_key = jax.random.fold_in(plan_key, int(self.plan_counts[env_index]))
        self.plan_counts[env_index] += 1
        return jax.random.fold_in(plan_key, 1), plan_key

    def _guidance_block(
        self,
        pixels,
        goals,
        key,
        temperature=0.0,
    ):
        observations = np.asarray(pixels[-1:])
        block = np.asarray(
            self.guidance_policy.sample_actions(
                observations=observations,
                goals=np.asarray(goals[-1:]),
                seed=key,
                temperature=temperature,
            )
        )
        if block.shape != (1, self.block_action_dim):
            raise ValueError(
                f'Guidance policy returned {block.shape}; expected '
                f'(1, {self.block_action_dim}).'
            )
        block = block[0]
        atomic = block.reshape(-1, self.atomic_action_dim)
        return self.scaler.transform(atomic).reshape(-1)

    def _guidance_population(
        self, pixels, goals, key
    ):
        count = self.guidance_population_size
        total_count = count * self.iterations
        sample_key, mode_key = jax.random.split(key)
        observations = np.repeat(
            np.asarray(pixels[-1:]), total_count, axis=0
        )
        blocks = np.asarray(
            self.guidance_policy.sample_actions(
                observations=observations,
                goals=np.repeat(np.asarray(goals[-1:]), total_count, axis=0),
                seed=sample_key,
                temperature=self.guidance_temperature,
            )
        )
        if blocks.shape != (total_count, self.block_action_dim):
            raise ValueError(
                f'Guidance population returned {blocks.shape}; expected '
                f'({total_count}, {self.block_action_dim}).'
            )
        atomic = blocks.reshape(-1, self.atomic_action_dim)
        blocks = self.scaler.transform(atomic).reshape(blocks.shape)
        blocks = blocks.reshape(self.iterations, count, self.block_action_dim).copy()
        mode = self._guidance_block(
            pixels,
            goals,
            mode_key,
            temperature=0.0,
        )
        blocks[:, 0] = mode
        return blocks

    def _initial_mean(
        self,
        env_index,
        pixels,
        goals,
        guidance_key,
        target_embedding=None,
        guidance_block=None,
    ):
        initial = np.zeros(
            (self.horizon, self.block_action_dim), dtype=np.float32
        )
        warm = self.warm_starts[env_index]
        if warm is not None:
            initial[: len(warm)] = warm
        if self.guidance_policy is not None:
            initial[0] = (
                guidance_block
                if guidance_block is not None
                else self._guidance_block(
                    pixels,
                    goals,
                    guidance_key,
                )
            )
        return initial

    def get_actions(self, pixels, goals, alive):
        for env_index in np.flatnonzero(alive):
            if self.subgoal_generator is not None:
                self.subgoal_generator.observe(
                    env_index, np.asarray(pixels[env_index, -1])
                )
            if self.buffers[env_index]:
                continue

            guidance_key, plan_key = self._next_plan_keys(env_index)
            if self.subgoal_generator is None:
                target_embedding = np.zeros(
                    int(self.lewm_config['embed_dim']), dtype=np.float32
                )
            else:
                predicted_path = self.subgoal_generator.predict_path(
                    env_index, np.asarray(goals[env_index, -1])
                )
                target_embedding = (
                    predicted_path
                    if self.cost_mode == 'path_mean'
                    else predicted_path[-1]
                )
            if self.guidance_mode == 'policy_random_mixture':
                guidance_blocks = self._guidance_population(
                    pixels[env_index],
                    goals[env_index],
                    guidance_key,
                )
                guidance_block = guidance_blocks[0, 0]
            else:
                guidance_blocks = np.zeros((1, self.block_action_dim), dtype=np.float32)
                guidance_block = None
            initial_mean = self._initial_mean(
                env_index,
                pixels[env_index],
                goals[env_index],
                guidance_key,
                target_embedding=target_embedding,
                guidance_block=guidance_block,
            )
            plan_output = self._plan_one(
                plan_key,
                jnp.asarray(pixels[env_index]),
                jnp.asarray(goals[env_index]),
                jnp.asarray(target_embedding),
                jnp.asarray(initial_mean),
                jnp.asarray(guidance_blocks),
            )
            normalized_blocks = plan_output[0]
            normalized_blocks = np.asarray(normalized_blocks)
            if self.subgoal_generator is not None and self.collect_traces:
                current_embedding = np.asarray(
                    self.encode_pixels(np.asarray(pixels[env_index, -1:]))[0],
                    dtype=np.float32,
                )
                imagined_path = np.asarray(
                    self._rollout_selected_plan(
                        jnp.asarray(pixels[env_index]),
                        jnp.asarray(goals[env_index]),
                        jnp.asarray(normalized_blocks),
                    ),
                    dtype=np.float32,
                )
                environment_action_blocks = self.scaler.inverse_transform(
                    normalized_blocks.reshape(-1, self.atomic_action_dim)
                ).reshape(
                    normalized_blocks.shape[0],
                    self.action_block,
                    self.atomic_action_dim,
                )
                trace = {
                        'environment_step': int(self.environment_steps[env_index]),
                        'current_embedding': current_embedding,
                        'predicted_path': predicted_path.copy(),
                        'imagined_path': imagined_path,
                        'normalized_action_blocks': normalized_blocks.copy(),
                        'environment_action_blocks': environment_action_blocks,
                    }
                if self.trace_candidates:
                    candidate_normalized = np.asarray(plan_output[2])
                    candidate_environment = self.scaler.inverse_transform(
                        candidate_normalized.reshape(-1, self.atomic_action_dim)
                    ).reshape(
                        candidate_normalized.shape[0],
                        candidate_normalized.shape[1],
                        self.action_block,
                        self.atomic_action_dim,
                    )
                    trace.update(
                        {
                            'candidate_normalized_action_blocks': candidate_normalized,
                            'candidate_environment_action_blocks': candidate_environment,
                            'candidate_imagined_paths': np.asarray(plan_output[3]),
                            'candidate_goal_costs': np.asarray(plan_output[4]),
                        }
                    )
                self.subgoal_traces[env_index].append(trace)
            keep = normalized_blocks[: self.receding_horizon]
            self.warm_starts[env_index] = normalized_blocks[
                self.receding_horizon:
            ].copy()
            normalized_atomic = keep.reshape(-1, self.atomic_action_dim)
            self.buffers[env_index].extend(
                self.scaler.inverse_transform(normalized_atomic)
            )

        actions = np.full(
            (len(alive), self.atomic_action_dim), np.nan, dtype=np.float32
        )
        for env_index in np.flatnonzero(alive):
            actions[env_index] = self.buffers[env_index].popleft()
            self.environment_steps[env_index] += 1
        return actions
