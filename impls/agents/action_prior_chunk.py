"""Final-goal-conditioned action-chunk prior with frozen LeWM representations."""

from __future__ import annotations

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue

REPRESENTATION_MODULES = ('q', 'v', 'pi')
REPRESENTATION_MODES = {
    'all': (True, True, True),
    'pi': (False, False, True),
    'v': (True, True, False),
}


def representation_sharing(mode):
    """Return the frozen-LeWM sharing flags for Q, V, and policy."""
    try:
        return dict(zip(REPRESENTATION_MODULES, REPRESENTATION_MODES[mode]))
    except KeyError as error:
        choices = ', '.join(REPRESENTATION_MODES)
        raise ValueError(f'Unsupported action-prior representation mode {mode!r}; choose {choices}.') from error


def resolve_representation_mode(representation, config):
    """Resolve a persisted mode and reject disagreement between metadata sections."""
    metadata_mode = representation.get('mode')
    config_mode = config.get('representation_mode')
    if metadata_mode is not None and config_mode is not None and metadata_mode != config_mode:
        raise ValueError(
            f'Action-prior representation metadata disagrees: representation.mode={metadata_mode!r}, '
            f'agent.representation_mode={config_mode!r}.'
        )
    mode = metadata_mode if metadata_mode is not None else config_mode
    representation_sharing(mode)
    return mode


def validate_representation_sharing(mode, config, label='config'):
    """Check that a mode and its three persisted sharing flags agree."""
    expected = representation_sharing(mode)
    actual = {module: bool(config.get(f'share_{module}_encoder', False)) for module in REPRESENTATION_MODULES}
    if actual != expected:
        raise ValueError(
            f'Action-prior representation {label} is inconsistent: mode={mode!r} expects {expected}, got {actual}.'
        )
    return expected


class ActionPriorChunkAgent(flax.struct.PyTreeNode):
    """Value-weighted action-chunk prior in a shared frozen LeWM space."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @property
    def action_horizon(self):
        return int(self.config['chunk_size'])

    @staticmethod
    def expectile_loss(advantage, difference, expectile):
        weight = jnp.where(advantage >= 0, expectile, 1 - expectile)
        return weight * difference**2

    def _inputs(self, batch, module, goal_kind, next_state=False):
        shared = self.config[f'share_{module}_encoder']
        prefix = 'lewm_' if shared else ''
        observation_key = 'next_observations' if next_state else 'observations'
        return batch[prefix + observation_key], batch[prefix + goal_kind + '_goals']

    def value_loss(self, batch, grad_params):
        q_observations, q_goals = self._inputs(batch, 'q', 'value')
        q1, q2 = self.network.select('target_critic')(q_observations, q_goals, batch['actions'])
        q = jnp.minimum(q1, q2)
        v_observations, v_goals = self._inputs(batch, 'v', 'value')
        v = self.network.select('value')(v_observations, v_goals, params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        next_v_observations, v_goals = self._inputs(batch, 'v', 'value', next_state=True)
        next_v = self.network.select('value')(next_v_observations, v_goals)
        chunk_discount = self.config['discount'] ** self.config['chunk_size']
        target = batch['rewards'] + chunk_discount * batch['masks'] * next_v

        q_observations, q_goals = self._inputs(batch, 'q', 'value')
        q1, q2 = self.network.select('critic')(
            q_observations,
            q_goals,
            batch['actions'],
            params=grad_params,
        )
        critic_loss = ((q1 - target) ** 2 + (q2 - target) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': target.mean(),
            'q_max': target.max(),
            'q_min': target.min(),
            'chunk_reward_mean': batch['rewards'].mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        del rng
        q_observations, q_goals = self._inputs(batch, 'q', 'actor')
        v_observations, v_goals = self._inputs(batch, 'v', 'actor')
        pi_observations, pi_goals = self._inputs(batch, 'pi', 'actor')

        v = self.network.select('value')(v_observations, v_goals)
        q1, q2 = self.network.select('critic')(q_observations, q_goals, batch['actions'])
        advantage = jnp.minimum(q1, q2) - v
        weights = jnp.minimum(jnp.exp(advantage * self.config['alpha']), 100.0)
        distribution = self.network.select('actor')(pi_observations, pi_goals, params=grad_params)
        log_probability = distribution.log_prob(batch['actions'])
        actor_loss = -(weights * log_probability).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': advantage.mean(),
            'bc_log_prob': log_probability.mean(),
            'mse': jnp.mean((distribution.mode() - batch['actions']) ** 2),
            'std': jnp.mean(distribution.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = self.rng if rng is None else rng
        value_loss, value_info = self.value_loss(batch, grad_params)
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info.update({f'value/{key}': value for key, value in value_info.items()})
        info.update({f'critic/{key}': value for key, value in critic_info.items()})
        info.update({f'actor/{key}': value for key, value in actor_info.items()})
        return value_loss + critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda parameter, target: parameter * self.config['tau'] + target * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        distribution = self.network.select('actor')(observations, goals, temperature=temperature)
        return jnp.clip(distribution.sample(seed=seed), -1, 1)

    @classmethod
    def create(cls, seed, ex_pixels, ex_latents, ex_actions, config):
        if config['discrete']:
            raise ValueError('ActionPriorChunkAgent supports continuous actions only.')
        validate_representation_sharing(config['representation_mode'], config)
        if config['encoder'] is None and not all(
            config[f'share_{module}_encoder'] for module in REPRESENTATION_MODULES
        ):
            raise ValueError('Non-shared modules require a pixel encoder.')

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        action_dim = ex_actions.shape[-1]

        def gc_encoder(module):
            if config[f'share_{module}_encoder']:
                return None
            return GCEncoder(concat_encoder=encoder_modules[config['encoder']]())

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=gc_encoder('v'),
        )
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=gc_encoder('q'),
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=gc_encoder('pi'),
        )

        def examples(module):
            observations = ex_latents if config[f'share_{module}_encoder'] else ex_pixels
            return observations, observations

        network_info = {
            'value': (value_def, examples('v')),
            'critic': (critic_def, (*examples('q'), ex_actions)),
            'target_critic': (copy.deepcopy(critic_def), (*examples('q'), ex_actions)),
            'actor': (actor_def, examples('pi')),
        }
        network_def = ModuleDict({name: definition for name, (definition, _) in network_info.items()})
        network_params = network_def.init(
            init_rng,
            **{name: args for name, (_, args) in network_info.items()},
        )['params']
        network_params['modules_target_critic'] = network_params['modules_critic']
        network = TrainState.create(
            network_def,
            network_params,
            tx=optax.adam(learning_rate=config['lr']),
        )
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='action_prior_chunk',
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            actor_loss='awr',
            alpha=3.0,
            const_std=True,
            discrete=False,
            encoder='impala_small',
            dataset_class='GCChunkDataset',
            chunk_size=5,
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=None,
            representation_mode='all',
            share_q_encoder=True,
            share_v_encoder=True,
            share_pi_encoder=True,
            latent_dim=192,
        )
    )
