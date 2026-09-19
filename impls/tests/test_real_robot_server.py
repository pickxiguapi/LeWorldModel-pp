import base64
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts.eval_real_robot_server import LeWMARXInferenceService, decode_infer_request


def encode_image(image, codec='jpg'):
    extension = '.png' if codec == 'png' else '.jpg'
    ok, encoded = cv2.imencode(extension, image)
    assert ok
    return {'codec': codec, 'data': base64.b64encode(encoded).decode('ascii')}


def payload(image, request_id=1):
    return {
        'request_id': request_id,
        'frames': [
            {'timestamp_ns': index + 1, 'image': encode_image(image)}
            for index in range(3)
        ],
        'goal': encode_image(image, codec='png'),
    }


class FakeLeWMPolicy:
    def __init__(self):
        self.reset_count = 0
        self.action_count = 0

    def warmup(self, observation, goal):
        assert observation.shape == goal.shape

    def reset(self, goal):
        self.reset_count += 1
        self.action_count = 0

    def act(self, observation):
        self.action_count += 1
        return np.full(14, self.action_count, dtype=np.float32)


class FakeLeWMPPPolicy:
    def warmup(self, observation, goal):
        assert observation.shape == goal.shape

    def plan_action_chunk(self, observations, goal):
        assert len(observations) == 3
        assert all(observation.shape == goal.shape for observation in observations)
        return np.arange(140, dtype=np.float32).reshape(10, 14)


def test_decode_unchanged_client_payload():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    request_id, frames, goal = decode_infer_request(payload(image), expected_hw=(4, 6))
    assert request_id == 1
    assert len(frames) == 3
    assert all(frame.shape == (4, 6, 3) for frame in frames)
    assert goal.shape == (4, 6, 3)


def test_decode_rejects_non_monotonic_timestamps():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    request = payload(image)
    request['frames'][1]['timestamp_ns'] = 1
    with pytest.raises(ValueError, match='strictly increasing'):
        decode_infer_request(request, expected_hw=(4, 6))


def test_lewm_server_returns_ten_actions_and_retains_episode_state():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    policy = FakeLeWMPolicy()
    service = LeWMARXInferenceService(
        policy,
        'lewm',
        {'lewm_checkpoint': '/checkpoint'},
        image_hw=(4, 6),
        warmup=False,
    )

    first = service.infer(payload(image, request_id=1))
    second = service.infer(payload(image, request_id=2))

    assert first['request_id'] == 1
    assert np.asarray(first['actions']).shape == (10, 14)
    assert policy.reset_count == 1
    np.testing.assert_array_equal(np.asarray(second['actions'])[:, 0], np.arange(11, 21))


@pytest.mark.parametrize('policy_name', ('lewmpp', 'lewmdp'))
def test_lewmpp_server_returns_exact_planned_chunk(policy_name):
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    service = LeWMARXInferenceService(
        FakeLeWMPPPolicy(),
        policy_name,
        {'lewm_checkpoint': '/checkpoint'},
        image_hw=(4, 6),
        warmup=False,
    )

    response = service.infer(payload(image))

    np.testing.assert_array_equal(
        np.asarray(response['actions'], dtype=np.float32),
        np.arange(140, dtype=np.float32).reshape(10, 14),
    )


@pytest.mark.parametrize('policy_name', ('lewm', 'lewmpp', 'lewmdp'))
def test_health_is_compatible_with_unchanged_client(policy_name):
    policy = FakeLeWMPolicy() if policy_name == 'lewm' else FakeLeWMPPPolicy()
    service = LeWMARXInferenceService(
        policy,
        policy_name,
        {'lewm_checkpoint': '/checkpoint'},
        image_hw=(480, 640),
        warmup=False,
    )
    health = service.health()
    assert health['camera_key'] == 'observation.images.camera_h'
    assert health['history_steps'] == 3
    assert health['image_hw'] == [480, 640]
    assert health['action_steps'] == 10
    assert health['action_dim'] == 14
    assert health['ddim_steps'] == 20
    assert health['legacy_ddim_field'] == (policy_name != 'lewmdp')
    if policy_name != 'lewm':
        expected_prior = 'diffusion_policy' if policy_name == 'lewmdp' else 'action_chunk_prior'
        assert health['action_prior'] == expected_prior
    if policy_name == 'lewmdp':
        assert health['policy_guidance'] == 'policy_random_mixture'
        assert health['cem_num_samples'] == 300
        assert health['cem_iterations'] == 2
        assert health['action_prior_population_size'] == 285
        assert health['random_population_size'] == 15


def test_health_reports_policy_best_of_n_without_cem_refits():
    policy = FakeLeWMPPPolicy()
    policy.controller = SimpleNamespace(
        horizon=1,
        receding_horizon=1,
        action_block=10,
        num_samples=256,
        iterations=1,
        action_prior_mode='policy_best_of_n',
        action_prior_population_size=256,
    )
    service = LeWMARXInferenceService(
        policy,
        'lewmdp',
        {'lewm_checkpoint': '/checkpoint'},
        image_hw=(480, 640),
        warmup=False,
    )

    health = service.health()

    assert health['cem_horizon'] == 1
    assert health['planning_horizon_actions'] == 10
    assert health['cem_num_samples'] == 256
    assert health['action_prior_population_size'] == 256
    assert health['random_population_size'] == 0
    assert health['candidate_evaluation_rounds'] == 1
    assert health['cem_refit_iterations'] == 0
