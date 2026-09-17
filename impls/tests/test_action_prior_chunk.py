import unittest

import jax.numpy as jnp
import pytest
from agents.action_prior_chunk import (
    REPRESENTATION_MODES,
    ActionPriorChunkAgent,
    get_config,
    representation_sharing,
)


class ActionPriorChunkAgentTest(unittest.TestCase):
    def test_public_representation_modes_have_exact_sharing_semantics(self):
        self.assertEqual(tuple(REPRESENTATION_MODES), ('all', 'pi', 'v'))
        self.assertEqual(representation_sharing('all'), {'q': True, 'v': True, 'pi': True})
        self.assertEqual(representation_sharing('pi'), {'q': False, 'v': False, 'pi': True})
        self.assertEqual(representation_sharing('v'), {'q': True, 'v': True, 'pi': False})
        with pytest.raises(ValueError, match='Unsupported action-prior representation mode'):
            representation_sharing('qv')

    def test_release_config_uses_five_action_chunks(self):
        config = get_config()
        self.assertEqual(config.agent_name, 'action_prior_chunk')
        self.assertEqual(config.chunk_size, 5)
        self.assertEqual(config.dataset_class, 'GCChunkDataset')
        self.assertEqual(config.representation_mode, 'all')
        self.assertTrue(config.share_q_encoder)
        self.assertTrue(config.share_v_encoder)
        self.assertTrue(config.share_pi_encoder)

    def make_agent_and_batch(self, shared):
        config = get_config()
        config.actor_hidden_dims = (8, 8)
        config.value_hidden_dims = (8, 8)
        config.chunk_size = 2
        config.latent_dim = 6
        config.encoder = 'impala_debug'
        config.representation_mode = {
            frozenset({'q', 'v', 'pi'}): 'all',
            frozenset({'pi'}): 'pi',
            frozenset({'q', 'v'}): 'v',
        }[frozenset(shared)]
        config.share_q_encoder = 'q' in shared
        config.share_v_encoder = 'v' in shared
        config.share_pi_encoder = 'pi' in shared

        pixels = jnp.zeros((2, 16, 16, 3), dtype=jnp.uint8)
        latents = jnp.zeros((2, config.latent_dim), dtype=jnp.float32)
        actions = jnp.zeros((2, 4), dtype=jnp.float32)
        agent = ActionPriorChunkAgent.create(0, pixels[:1], latents[:1], actions[:1], config)
        batch = {
            'observations': pixels,
            'next_observations': pixels,
            'value_goals': pixels,
            'actor_goals': pixels,
            'lewm_observations': latents,
            'lewm_next_observations': latents,
            'lewm_value_goals': latents,
            'lewm_actor_goals': latents,
            'actions': actions,
            'rewards': -jnp.ones((2,), dtype=jnp.float32),
            'masks': jnp.ones((2,), dtype=jnp.float32),
        }
        return agent, batch

    def test_requested_sharing_variants_have_finite_losses(self):
        for shared in ({'pi'}, {'q', 'v', 'pi'}, {'q', 'v'}):
            with self.subTest(shared=shared):
                agent, batch = self.make_agent_and_batch(shared)
                loss, info = agent.total_loss(batch, grad_params=None)
                self.assertTrue(jnp.isfinite(loss))
                self.assertIn('actor/actor_loss', info)

    def test_all_shared_agent_updates_and_emits_chunk(self):
        agent, batch = self.make_agent_and_batch({'q', 'v', 'pi'})
        agent, info = agent.update(batch)
        actions = agent.sample_actions(
            batch['lewm_observations'][:1],
            batch['lewm_actor_goals'][:1],
            seed=agent.rng,
            temperature=0.0,
        )
        self.assertEqual(actions.shape, (1, 4))
        self.assertTrue(jnp.isfinite(info['actor/actor_loss']))

    def test_lewm_inputs_are_required_only_by_shared_modules(self):
        agent, batch = self.make_agent_and_batch({'pi'})
        del batch['lewm_value_goals']
        # Q and V are pixel modules, so actor-goal LeWM inputs are sufficient.
        loss, _ = agent.total_loss(batch, grad_params=None)
        self.assertTrue(jnp.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
