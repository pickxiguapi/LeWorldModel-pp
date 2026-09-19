#!/usr/bin/env python3
"""HTTP inference server compatible with the existing GC-DP ARX client."""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CAMERA_KEY = 'observation.images.camera_h'
ACTION_ORDER = 'left_joint0..5,left_gripper,right_joint0..5,right_gripper'
MAX_REQUEST_BYTES = 16 * 1024 * 1024
WIRE_IMAGE_HW = (480, 640)


def _decode_image(payload: dict[str, str], expected_hw: tuple[int, int]) -> np.ndarray:
    if payload.get('codec') not in {'jpg', 'png'}:
        raise ValueError('Image codec must be jpg or png')
    encoded = base64.b64decode(payload['data'], validate=True)
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (*expected_hw, 3) or image.dtype != np.uint8:
        raise ValueError(f'Image must be BGR uint8 with shape {(*expected_hw, 3)}')
    return image


def decode_infer_request(
    payload: dict[str, Any],
    expected_hw: tuple[int, int] = WIRE_IMAGE_HW,
) -> tuple[int, list[np.ndarray], np.ndarray]:
    """Decode the unchanged GC-DP client payload into BGR image arrays."""
    frames = payload.get('frames')
    if not isinstance(frames, list) or len(frames) != 3:
        raise ValueError('Expected exactly three recent camera frames')
    timestamps = [int(frame['timestamp_ns']) for frame in frames]
    if not timestamps[0] < timestamps[1] < timestamps[2]:
        raise ValueError('Frame timestamps must be strictly increasing')
    request_id = int(payload['request_id'])
    if request_id <= 0:
        raise ValueError('request_id must be positive')
    images = [_decode_image(frame['image'], expected_hw) for frame in frames]
    goal = _decode_image(payload['goal'], expected_hw)
    return request_id, images, goal


class LeWMARXInferenceService:
    """Serialize access to one stateful LeWM or LeWM++ real-robot policy."""

    def __init__(
        self,
        policy,
        policy_name: str,
        checkpoint_paths: dict[str, str],
        *,
        image_hw: tuple[int, int] = WIRE_IMAGE_HW,
        warmup: bool = True,
    ) -> None:
        if policy_name not in {'lewm', 'lewmpp', 'lewmdp'}:
            raise ValueError(f'Unknown policy: {policy_name!r}')
        self.policy = policy
        self.policy_name = policy_name
        self.checkpoint_paths = checkpoint_paths
        self.image_hw = tuple(int(value) for value in image_hw)
        self.lock = threading.Lock()
        self.last_request_id = 0
        self.current_goal = None
        if warmup:
            image = np.zeros((*self.image_hw, 3), dtype=np.uint8)
            self.policy.warmup(image, image)
            self.current_goal = None
            self.last_request_id = 0

    def health(self) -> dict[str, Any]:
        # The unchanged legacy client requires this exact field. It is a
        # compatibility placeholder except when the prior is Diffusion Policy.
        result = {
            'ready': True,
            'policy': self.policy_name,
            'camera_key': CAMERA_KEY,
            'image_hw': list(self.image_hw),
            'history_steps': 3,
            'goal_steps': 1,
            'action_steps': 10,
            'action_dim': 14,
            'ddim_steps': 20,
            'legacy_ddim_field': self.policy_name != 'lewmdp',
            'action_type': 'absolute_joint',
            'action_order': ACTION_ORDER,
        }
        result.update(self.checkpoint_paths)
        if self.policy_name == 'lewm':
            result.update(
                {
                    'cem_horizon': 5,
                    'cem_receding_horizon': 5,
                    'action_block': 10,
                    'cem_num_samples': 300,
                    'cem_iterations': 30,
                }
            )
        else:
            controller = getattr(self.policy, 'controller', None)
            cem_num_samples = int(getattr(controller, 'num_samples', 300))
            cem_iterations = int(getattr(controller, 'iterations', 2 if self.policy_name == 'lewmdp' else 5))
            result.update(
                {
                    'cem_horizon': int(getattr(controller, 'horizon', 2)),
                    'cem_receding_horizon': int(getattr(controller, 'receding_horizon', 1)),
                    'action_block': int(getattr(controller, 'action_block', 10)),
                    'flow_sampling_steps': 16,
                    'action_prior': 'diffusion_policy' if self.policy_name == 'lewmdp' else 'action_chunk_prior',
                    'cem_num_samples': cem_num_samples,
                    'cem_iterations': cem_iterations,
                }
            )
            if self.policy_name == 'lewmdp':
                policy_guidance = str(getattr(controller, 'action_prior_mode', 'policy_random_mixture'))
                policy_population = int(getattr(controller, 'action_prior_population_size', 285))
                result.update(
                    {
                        'policy_guidance': policy_guidance,
                        'action_prior_population_size': policy_population,
                        'random_population_size': cem_num_samples - policy_population,
                        'candidate_evaluation_rounds': 1 if policy_guidance == 'policy_best_of_n' else cem_iterations,
                        'cem_refit_iterations': 0 if policy_guidance == 'policy_best_of_n' else cem_iterations,
                        'planning_horizon_actions': int(getattr(controller, 'horizon', 2))
                        * int(getattr(controller, 'action_block', 10)),
                    }
                )
        return result

    def _goal_changed(self, goal: np.ndarray) -> bool:
        return self.current_goal is None or not np.array_equal(goal, self.current_goal)

    def _infer_lewm(self, frames: list[np.ndarray], goal: np.ndarray, request_id: int) -> np.ndarray:
        if request_id <= self.last_request_id or self._goal_changed(goal):
            self.policy.reset(goal)
        # The official LeWM baseline retains H=5/RH=5, hence 50 buffered
        # atomic actions. The unchanged ARX wire protocol consumes ten per
        # request; new observations are used when that 50-action buffer empties.
        return np.stack([self.policy.act(frames[-1]) for _ in range(10)])

    def _infer_lewmpp(self, frames: list[np.ndarray], goal: np.ndarray) -> np.ndarray:
        return self.policy.plan_action_chunk(frames, goal)

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id, frames, goal = decode_infer_request(payload, self.image_hw)
        started = time.perf_counter()
        with self.lock:
            if self.policy_name == 'lewm':
                actions = self._infer_lewm(frames, goal, request_id)
            else:
                actions = self._infer_lewmpp(frames, goal)
            self.current_goal = goal.copy()
            self.last_request_id = request_id
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (10, 14) or not np.isfinite(actions).all():
            raise RuntimeError(f'Policy returned invalid actions: {actions.shape}')
        return {
            'request_id': request_id,
            'actions': actions.tolist(),
            'latency_ms': (time.perf_counter() - started) * 1000,
        }


class Handler(BaseHTTPRequestHandler):
    service: LeWMARXInferenceService

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != '/health':
            self._send(404, {'error': 'not found'})
            return
        self._send(200, self.service.health())

    def do_POST(self) -> None:
        if self.path != '/infer':
            self._send(404, {'error': 'not found'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= MAX_REQUEST_BYTES:
                raise ValueError('Invalid request length')
            payload = json.loads(self.rfile.read(size))
            response = self.service.infer(payload)
        except (ValueError, KeyError, TypeError) as exc:
            self._send(400, {'error': str(exc)})
            return
        except Exception as exc:
            self._send(500, {'error': f'Inference failed: {exc}'})
            return
        self._send(200, response)

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f'{self.client_address[0]} - {format_string % args}', flush=True)


def _resolved(path: Path) -> str:
    return str(path.expanduser().resolve())


def create_service(args: argparse.Namespace) -> LeWMARXInferenceService:
    if args.policy == 'lewm':
        from eval_real_robot_lewm import RealRobotLeWMPolicy

        policy = RealRobotLeWMPolicy(
            lance_path=args.lance_path,
            lewm_checkpoint=args.lewm_checkpoint,
            seed=args.seed,
            input_color='bgr',
        )
        paths = {'lewm_checkpoint': _resolved(args.lewm_checkpoint)}
    elif args.policy == 'lewmpp':
        if args.action_prior_dir is None or args.latent_path_flow_checkpoint is None:
            raise ValueError('LeWM++ requires --action-prior-dir and --latent-path-flow-checkpoint')
        from eval_real_robot_lewmpp import RealRobotLeWMPPPolicy

        policy = RealRobotLeWMPPPolicy(
            lance_path=args.lance_path,
            lewm_checkpoint=args.lewm_checkpoint,
            action_prior_dir=args.action_prior_dir,
            action_prior_step=args.action_prior_step,
            latent_path_flow_checkpoint=args.latent_path_flow_checkpoint,
            seed=args.seed,
            input_color='bgr',
        )
        paths = {
            'lewm_checkpoint': _resolved(args.lewm_checkpoint),
            'action_prior_dir': _resolved(args.action_prior_dir),
            'action_prior_step': int(args.action_prior_step),
            'latent_path_flow_checkpoint': _resolved(args.latent_path_flow_checkpoint),
        }
    else:
        if args.diffusion_policy_checkpoint is None or args.latent_path_flow_checkpoint is None:
            raise ValueError('LeWM-DP requires --diffusion-policy-checkpoint and --latent-path-flow-checkpoint')
        from eval_real_robot_lewmdp import RealRobotLeWMDPPolicy

        cem_iterations = args.cem_iterations
        if cem_iterations is None:
            cem_iterations = 1 if args.policy_guidance == 'policy_best_of_n' else 2
        policy = RealRobotLeWMDPPolicy(
            lance_path=args.lance_path,
            lewm_checkpoint=args.lewm_checkpoint,
            diffusion_policy_checkpoint=args.diffusion_policy_checkpoint,
            latent_path_flow_checkpoint=args.latent_path_flow_checkpoint,
            diffusion_device=args.diffusion_device,
            diffusion_batch_size=args.diffusion_batch_size,
            cem_num_samples=args.cem_num_samples,
            cem_iterations=cem_iterations,
            action_prior_population_size=args.diffusion_population_size,
            policy_guidance=args.policy_guidance,
            seed=args.seed,
            input_color='bgr',
        )
        paths = {
            'lewm_checkpoint': _resolved(args.lewm_checkpoint),
            'diffusion_policy_checkpoint': _resolved(args.diffusion_policy_checkpoint),
            'latent_path_flow_checkpoint': _resolved(args.latent_path_flow_checkpoint),
        }
    return LeWMARXInferenceService(
        policy,
        args.policy,
        paths,
        warmup=not args.no_warmup,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=('lewm', 'lewmpp', 'lewmdp'), required=True)
    parser.add_argument('--lance-path', type=Path, required=True)
    parser.add_argument('--lewm-checkpoint', type=Path, required=True)
    parser.add_argument('--action-prior-dir', type=Path)
    parser.add_argument('--action-prior-step', type=int, default=100_000)
    parser.add_argument('--latent-path-flow-checkpoint', type=Path)
    parser.add_argument('--diffusion-policy-checkpoint', type=Path)
    parser.add_argument('--diffusion-device', default='cuda:0')
    parser.add_argument('--diffusion-batch-size', type=int, default=32)
    parser.add_argument(
        '--policy-guidance',
        choices=('policy_random_mixture', 'policy_best_of_n'),
        default='policy_random_mixture',
    )
    parser.add_argument('--cem-num-samples', type=int, default=300)
    parser.add_argument('--cem-iterations', type=int)
    parser.add_argument('--diffusion-population-size', type=int, default=285)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-warmup', action='store_true')
    args = parser.parse_args()
    if (
        args.action_prior_step <= 0
        or args.diffusion_batch_size <= 0
        or args.cem_num_samples <= 1
        or args.diffusion_population_size <= 0
        or (args.cem_iterations is not None and args.cem_iterations <= 0)
        or not 0 < args.port < 65536
    ):
        parser.error('Checkpoint steps, population sizes, iterations, batch size, and port must be positive')
    return args


def main() -> None:
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    service = create_service(args)
    handler = type('LeWMARXHandler', (Handler,), {'service': service})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f'{args.policy} ready at http://{args.host}:{args.port}; '
        f'lewm_checkpoint={service.checkpoint_paths["lewm_checkpoint"]}',
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
