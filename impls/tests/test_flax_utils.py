from pathlib import Path

import numpy as np
from utils.flax_utils import restore_agent, save_agent


def test_restore_agent_accepts_pathlib_directory(tmp_path: Path):
    saved = {'weight': np.asarray([1.0, 2.0])}
    template = {'weight': np.zeros(2)}
    save_agent(saved, tmp_path, 7)

    restored = restore_agent(template, tmp_path, 7)

    np.testing.assert_array_equal(restored['weight'], saved['weight'])
