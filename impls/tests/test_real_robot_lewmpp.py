from collections import deque

import numpy as np
from eval_real_robot_lewm import RealRobotLeWMPolicy, extract_camera_frame
from eval_real_robot_lewmpp import RealRobotLeWMPPPolicy
from real_robot_lewmpp import PUSH_MULTI_RED_CUBE_ACTION_NAMES, preprocess_robot_image


class FakeController:
    def __init__(self, action_count=10):
        self.action_count = action_count
        self.buffers = [deque()]
        self.subgoal_generator = type('Generator', (), {'observe': lambda self, index, pixels: None})()

    def reset(self, action_space, num_envs):
        assert action_space.shape == (14,)
        assert num_envs == 1
        self.buffers = [deque()]

    def get_actions(self, pixels, goals, alive):
        assert pixels.shape == (1, 1, 168, 224, 3)
        assert goals.shape == (1, 1, 168, 224, 3)
        np.testing.assert_array_equal(alive, [True])
        chunk = np.arange(self.action_count * 14, dtype=np.float32).reshape(self.action_count, 14)
        self.buffers[0].extend(chunk[1:])
        return chunk[:1]


def test_real_robot_preprocessing_matches_training_long_edge_resize():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[..., 0] = 255
    pixels = preprocess_robot_image(frame, (168, 224, 3), input_color='rgb')
    assert pixels.shape == (168, 224, 3)
    np.testing.assert_array_equal(pixels[..., 0], 255)
    np.testing.assert_array_equal(pixels[..., 1:], 0)


def test_real_robot_preprocessing_supports_opencv_bgr():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[..., 2] = 255
    pixels = preprocess_robot_image(frame, (168, 224, 3), input_color='bgr')
    np.testing.assert_array_equal(pixels[..., 0], 255)
    np.testing.assert_array_equal(pixels[..., 1:], 0)


def test_camera_frame_can_be_read_from_robot_observation_mapping():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert extract_camera_frame({'observation.images.camera_h': frame}) is frame
    assert extract_camera_frame({'observation': {'images': {'camera_h': frame}}}) is frame


def test_several_frames_produce_one_ten_action_chunk():
    policy = object.__new__(RealRobotLeWMPPPolicy)
    policy.input_color = 'rgb'
    policy.image_shape = (168, 224, 3)
    policy.action_dim = 14
    policy.action_low = np.full(14, -1_000.0, dtype=np.float32)
    policy.action_high = np.full(14, 1_000.0, dtype=np.float32)
    policy.action_chunk_size = 10
    policy.goal_pixels = None
    policy.controller = FakeController()
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
    goal = np.zeros((480, 640, 3), dtype=np.uint8)
    chunk = policy.plan_action_chunk(frames, goal)
    assert chunk.shape == (10, 14)
    np.testing.assert_array_equal(chunk, np.arange(140, dtype=np.float32).reshape(10, 14))


def test_lewm_baseline_observation_produces_official_receding_horizon_chunk():
    policy = object.__new__(RealRobotLeWMPolicy)
    policy.input_color = 'rgb'
    policy.image_shape = (168, 224, 3)
    policy.action_dim = 14
    policy.action_low = np.full(14, -1_000.0, dtype=np.float32)
    policy.action_high = np.full(14, 1_000.0, dtype=np.float32)
    policy.action_chunk_size = 50
    policy.goal_pixels = None
    policy.controller = FakeController(action_count=50)
    observation = {'camera_h': np.zeros((480, 640, 3), dtype=np.uint8)}
    goal = {'camera_h': np.zeros((480, 640, 3), dtype=np.uint8)}

    chunk = policy.plan_action_chunk(observation, goal)

    assert chunk.shape == (50, 14)
    np.testing.assert_array_equal(chunk, np.arange(700, dtype=np.float32).reshape(50, 14))


def test_real_robot_action_order_matches_lerobot_dataset():
    assert PUSH_MULTI_RED_CUBE_ACTION_NAMES == (
        'left_joint_0',
        'left_joint_1',
        'left_joint_2',
        'left_joint_3',
        'left_joint_4',
        'left_joint_5',
        'left_gripper',
        'right_joint_0',
        'right_joint_1',
        'right_joint_2',
        'right_joint_3',
        'right_joint_4',
        'right_joint_5',
        'right_gripper',
    )
