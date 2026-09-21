"""GCIQL-Chunk with independently selectable frozen LeWM representations."""

from __future__ import annotations

import copy

import flax
import jax
import jax.numpy as jnp
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCActor, GCValue

from agents.action_prior_chunk_ogbench import GCIQLChunkAgent
from agents.action_prior_chunk_ogbench import get_config as get_gciql_chunk_config


class LeWMGCIQLChunkAgent(GCIQLChunkAgent):
    """GCIQL-Chunk whose Q, V, and actor may independently consume LeWM latents.

    The frozen LeWM encoder is deliberately kept outside this train state.  A
    training batch contains both pixels and ``lewm_*`` latents, and each module
    selects its input according to its own sharing flag.  Thus all enabled
    modules use exactly the same frozen LeWM coordinate system while retaining
    separate downstream heads.
    """

    def _inputs(self, batch, module, goal_kind, next_state=False):
        shared = self.config[f'share_{module}_encoder']
        prefix = 'lewm_' if shared else ''
        observation_key = 'next_observations' if next_state else 'observations'
        return batch[prefix + observation_key], batch[prefix + goal_kind + '_goals']

    def value_loss(self, batch, grad_params):
        q_observations, q_goals = self._inputs(batch, 'q', 'value')
        q1, q2 = self.network.select('target_critic')(
            q_observations, q_goals, batch['actions']
        )
        q = jnp.minimum(q1, q2)
        v_observations, v_goals = self._inputs(batch, 'v', 'value')
        v = self.network.select('value')(
            v_observations, v_goals, params=grad_params
        )
        value_loss = self.expectile_loss(
            q - v, q - v, self.config['expectile']
        ).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        next_v_observations, v_goals = self._inputs(
            batch, 'v', 'value', next_state=True
        )
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
        q_observations, q_goals = self._inputs(batch, 'q', 'actor')
        v_observations, v_goals = self._inputs(batch, 'v', 'actor')
        pi_observations, pi_goals = self._inputs(batch, 'pi', 'actor')

        if self.config['actor_loss'] == 'awr':
            v = self.network.select('value')(v_observations, v_goals)
            q1, q2 = self.network.select('critic')(
                q_observations, q_goals, batch['actions']
            )
            adv = jnp.minimum(q1, q2) - v
            weights = jnp.minimum(jnp.exp(adv * self.config['alpha']), 100.0)
            dist = self.network.select('actor')(
                pi_observations, pi_goals, params=grad_params
            )
            log_prob = dist.log_prob(batch['actions'])
            actor_loss = -(weights * log_prob).mean()
            return actor_loss, {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }

        if self.config['actor_loss'] == 'ddpgbc':
            dist = self.network.select('actor')(
                pi_observations, pi_goals, params=grad_params
            )
            q_actions = (
                dist.mode()
                if self.config['const_std']
                else dist.sample(seed=rng)
            )
            q_actions = jnp.clip(q_actions, -1, 1)
            q1, q2 = self.network.select('critic')(
                q_observations, q_goals, q_actions
            )
            q = jnp.minimum(q1, q2)
            q_loss = -q.mean() / jax.lax.stop_gradient(
                jnp.abs(q).mean() + 1e-6
            )
            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -(self.config['alpha'] * log_prob).mean()
            actor_loss = q_loss + bc_loss
            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }

        raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @classmethod
    def create(cls, seed, ex_pixels, ex_latents, ex_actions, config):
        if config['discrete']:
            raise ValueError('LeWMGCIQLChunkAgent supports continuous actions only.')
        if config['encoder'] is None and not all(
            config[f'share_{module}_encoder'] for module in ('q', 'v', 'pi')
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
            observations = (
                ex_latents if config[f'share_{module}_encoder'] else ex_pixels
            )
            return observations, observations

        network_info = {
            'value': (value_def, examples('v')),
            'critic': (critic_def, (*examples('q'), ex_actions)),
            'target_critic': (
                copy.deepcopy(critic_def),
                (*examples('q'), ex_actions),
            ),
            'actor': (actor_def, examples('pi')),
        }
        network_def = ModuleDict(
            {name: definition for name, (definition, _) in network_info.items()}
        )
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
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = get_gciql_chunk_config()
    config.agent_name = 'gciql_chunk_lewm'
    config.actor_loss = 'awr'
    config.alpha = 3.0
    config.encoder = 'impala_small'
    config.share_q_encoder = False
    config.share_v_encoder = False
    config.share_pi_encoder = False
    config.latent_dim = 192
    return config
