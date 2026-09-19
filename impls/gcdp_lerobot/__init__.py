"""Compatibility loader for the goal-conditioned Diffusion Policy checkpoint.

The model implementation is derived from LeRobot at commit e86f5af5 with the
goal-conditioning changes used to train the released GC-DP checkpoint. LeRobot
is licensed under Apache-2.0; the vendored source files retain their notices.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import draccus

from gcdp_lerobot.configuration_diffusion import DiffusionConfig
from gcdp_lerobot.modeling_diffusion import DiffusionPolicy


def load_goal_conditioned_diffusion_policy(checkpoint, device):
    """Load the exact GC-DP architecture without modifying installed LeRobot."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    config_path = checkpoint / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    raw = json.loads(config_path.read_text())
    if raw.pop('type', None) != 'diffusion':
        raise ValueError(f'Expected a Diffusion Policy config at {config_path}.')
    with tempfile.NamedTemporaryFile('w', suffix='.json') as file:
        json.dump(raw, file)
        file.flush()
        with draccus.config_type('json'):
            config = draccus.parse(DiffusionConfig, file.name, args=[])
    config.device = str(device)
    return DiffusionPolicy.from_pretrained(
        str(checkpoint),
        config=config,
        strict=True,
    ).to(device).eval()


__all__ = ['DiffusionConfig', 'DiffusionPolicy', 'load_goal_conditioned_diffusion_policy']
