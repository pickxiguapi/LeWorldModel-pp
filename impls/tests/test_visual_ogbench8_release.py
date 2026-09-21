import json
import subprocess
import sys
from pathlib import Path

import pytest
from action_prior_runtime_ogbench import load_agent_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('agent_name', ('gciql_chunk', 'gciql_chunk_lewm'))
def test_public_action_prior_names_restore_their_configs(tmp_path, agent_name):
    checkpoint_dir = tmp_path / agent_name
    checkpoint_dir.mkdir()
    (checkpoint_dir / 'flags.json').write_text(
        json.dumps({'agent': {'agent_name': agent_name, 'chunk_size': 5}})
    )
    restored_name, config, _ = load_agent_config(checkpoint_dir)
    assert restored_name == agent_name
    assert config.agent_name == agent_name
    assert config.chunk_size == 5


def test_visual_ogbench8_pipeline_is_path_connected():
    names = (
        'train_lewm_visual_ogbench8.sh',
        'precompute_visual_ogbench8_latents.sh',
        'train_action-prior-chunk_visual_ogbench8.sh',
        'train_latent_path_flow_visual_ogbench8.sh',
        'eval_lewmpp_visual_ogbench8.sh',
    )
    scripts = {
        name: (
            ROOT
            / 'experiments'
            / ('train' if name.startswith('train_') else 'eval' if name.startswith('eval_') else '')
            / name
        ).read_text()
        for name in names
    }
    assert 'weights_epoch_10.msgpack' in scripts['train_lewm_visual_ogbench8.sh'] or '--epochs=10' in scripts[
        'train_lewm_visual_ogbench8.sh'
    ]
    for name in scripts:
        if name.startswith('eval_'):
            assert '${GPU_IDS:-0 1 2 3 4 5 6 7}' in scripts[name]
        else:
            assert 'GPU_IDS=(0 1 2 3 4 5 6 7)' in scripts[name]
    assert 'LEWM_CHECKPOINT_ROOT=""' in scripts['eval_lewmpp_visual_ogbench8.sh']
    assert 'ACTION_PRIOR_CHECKPOINT_ROOT=""' in scripts['eval_lewmpp_visual_ogbench8.sh']
    assert 'LATENT_PATH_FLOW_CHECKPOINT_ROOT=""' in scripts['eval_lewmpp_visual_ogbench8.sh']
    assert 'checkpoint_200000.msgpack' in scripts['eval_lewmpp_visual_ogbench8.sh']
    assert '--policy-checkpoint-step=500000' in scripts['eval_lewmpp_visual_ogbench8.sh']


def test_visual_ogbench8_eval_records_paper_protocol():
    text = (ROOT / 'experiments' / 'eval' / 'eval_lewmpp_visual_ogbench8.sh').read_text()
    for fragment in (
        '${EVAL_SEEDS:-0 1 42}',
        'NUM_EVAL=${NUM_EVAL:-50}',
        '--num-eval="$NUM_EVAL"',
        '--cem-horizon=2',
        '--cem-receding-horizon=1',
        '--action-block=5',
        '--cem-num-samples=300',
        '--cem-iterations=5',
        '--cem-topk=30',
        '--guidance-population-size=250',
        '--guidance-random-elite-cap=5',
        '--cem-cost-mode=moh',
    ):
        assert fragment in text


def test_visual_ogbench8_eval_smoke_overrides_do_not_change_paper_defaults():
    eval_root = ROOT / 'experiments' / 'eval'
    for name in ('eval_lewmpp_visual_ogbench8.sh', 'eval_lewm_baseline_visual_ogbench8.sh'):
        text = (eval_root / name).read_text()
        assert '${GPU_IDS:-0 1 2 3 4 5 6 7}' in text
        assert '${EVAL_SEEDS:-0 1 42}' in text
        assert 'NUM_EVAL=${NUM_EVAL:-50}' in text
        assert '"$NUM_EVAL" == 50' in text


def test_visual_ogbench8_aggregator_rejects_incomplete_matrix(tmp_path):
    output = tmp_path / 'summary.json'
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'impls' / 'aggregate_visual_ogbench_results.py'),
            f'--results-root={tmp_path}',
            f'--output={output}',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'Incomplete matrix' in result.stderr
    assert not output.exists()


def test_visual_ogbench8_aggregator_uses_sample_standard_deviation(tmp_path):
    tags = ('cs_play', 'cd_play', 'ct_play', 'scene_play', 'cs_noisy', 'cd_noisy', 'ct_noisy', 'scene_noisy')
    for seed, value in zip((0, 1, 42), (0.4, 0.5, 0.6)):
        for tag in tags:
            path = tmp_path / tag / f'seed{seed}' / 'result.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        'seed': seed,
                        'episodes_per_task': 50,
                        'use_subgoal': True,
                        'policy_guidance': 'policy_random_mixture',
                        'overall_success': value,
                    }
                )
            )
    output = tmp_path / 'summary.json'
    subprocess.run(
        [
            sys.executable,
            str(ROOT / 'impls' / 'aggregate_visual_ogbench_results.py'),
            f'--results-root={tmp_path}',
            f'--output={output}',
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary['macro']['mean'] == 0.5
    assert summary['macro']['sample_std'] == pytest.approx(0.1)
