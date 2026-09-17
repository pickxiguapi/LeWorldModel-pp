import h5py
import numpy as np
from precompute_lewm_lance_latents import compute_episode_layout, copy_non_pixel_hdf5


def test_compute_episode_layout():
    offsets, lengths = compute_episode_layout([7, 7, 9, 9, 9, 4])
    np.testing.assert_array_equal(offsets, [0, 2, 5])
    np.testing.assert_array_equal(lengths, [2, 3, 1])


def test_copy_non_pixel_hdf5_preserves_metadata_and_excludes_images(tmp_path):
    source_path = tmp_path / 'source.h5'
    destination_path = tmp_path / 'destination.h5'
    with h5py.File(source_path, 'w') as source:
        source.attrs['name'] = 'tiny'
        source.create_dataset('pixels', data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
        action = source.create_dataset('action', data=np.asarray([[1.0], [np.nan], [3.0]], dtype=np.float32))
        action.attrs['units'] = 'normalized'
        nested = source.create_group('nested')
        nested.create_dataset('pixels', data=np.ones((3, 2, 2, 3), dtype=np.uint8))
        nested.create_dataset('state', data=np.arange(6).reshape(3, 2))

    with h5py.File(destination_path, 'w') as destination:
        copy_non_pixel_hdf5(source_path, destination)

    with h5py.File(destination_path, 'r') as destination:
        assert destination.attrs['name'] == 'tiny'
        assert 'pixels' not in destination
        assert 'pixels' not in destination['nested']
        np.testing.assert_allclose(destination['action'][...], [[1.0], [np.nan], [3.0]], equal_nan=True)
        assert destination['action'].attrs['units'] == 'normalized'
        np.testing.assert_array_equal(destination['nested/state'][...], np.arange(6).reshape(3, 2))
