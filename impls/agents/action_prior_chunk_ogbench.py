"""Goal-conditioned IQL in a fixed-length Gaussian action-chunk space."""

from agents.gciql import GCIQLAgent
from agents.gciql import get_config as get_gciql_config


class GCIQLChunkAgent(GCIQLAgent):
    """Flat GCIQL with a chunk-conditioned critic and Gaussian DDPG+BC actor.

    The actor and critic operate on flattened length-k action sequences.  The
    dataset supplies the exact discounted reward accumulated by the sequence,
    so the critic bootstraps from ``s[t + k]`` with discount ``gamma ** k``.
    """

    @property
    def action_horizon(self):
        """Number of atomic actions returned by one policy invocation."""
        return int(self.config['chunk_size'])

    def critic_loss(self, batch, grad_params):
        """Compute the k-step IQL critic loss for a full action chunk."""
        next_v = self.network.select('value')(batch['next_observations'], batch['value_goals'])
        chunk_discount = self.config['discount'] ** self.config['chunk_size']
        q = batch['rewards'] + chunk_discount * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'chunk_reward_mean': batch['rewards'].mean(),
        }

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a continuous-action Gaussian GCIQL-Chunk agent."""
        if config['discrete']:
            raise ValueError('GCIQLChunkAgent currently supports continuous action spaces only.')
        return super().create(seed, ex_observations, ex_actions, config)


def get_config():
    """Return GCIQL defaults with action chunking enabled."""
    config = get_gciql_config()
    config.agent_name = 'gciql_chunk'
    config.dataset_class = 'GCChunkDataset'
    config.chunk_size = 5
    config.actor_loss = 'ddpgbc'
    config.alpha = 1.0
    config.discrete = False
    return config
