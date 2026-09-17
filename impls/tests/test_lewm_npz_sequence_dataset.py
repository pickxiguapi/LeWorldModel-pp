import numpy as np
from utils.lewm_npz_sequence_dataset import LeWMNPZSequenceDataset


def _write_split(path, episode_lengths):
    total = sum(episode_lengths)
    observations = np.arange(total * 8 * 8 * 3, dtype=np.uint8).reshape(total, 8, 8, 3)
    actions = np.arange(total * 2, dtype=np.float32).reshape(total, 2)
    terminals = np.zeros(total, dtype=np.float32)
    terminals[np.cumsum(episode_lengths) - 1] = 1.0
    np.savez(path, observations=observations, actions=actions, terminals=terminals)


def test_npz_sequence_dataset_preserves_episode_boundaries(tmp_path):
    train_path = tmp_path / 'visual-cube-single-play-v0.npz'
    val_path = tmp_path / 'visual-cube-single-play-v0-val.npz'
    _write_split(train_path, [6, 5])
    _write_split(val_path, [4])

    dataset = LeWMNPZSequenceDataset(
        train_path,
        val_path,
        num_steps=4,
        frameskip=1,
        seed=0,
    )
    assert len(dataset.train_indices) == 5
    assert len(dataset.val_indices) == 1

    batch = dataset.get_batch(dataset.train_indices[[0, 3]])
    assert batch['pixels'].shape == (2, 4, 8, 8, 3)
    assert batch['action'].shape == (2, 4, 2)
    assert batch['pixels'].dtype == np.uint8
    assert batch['action'].dtype == np.float32

    val_batch = dataset.get_batch(dataset.val_indices)
    assert val_batch['pixels'].shape == (1, 4, 8, 8, 3)
    dataset.close()
