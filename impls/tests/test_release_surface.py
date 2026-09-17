import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from eval_lewm_control_suite import (
    DEFAULT_CEM_ITERATIONS,
    DEFAULT_CEM_SAMPLES,
    DEFAULT_FLOW_STEPS,
    VARIANTS,
    expected_components,
    validate_args,
    validate_release_files,
)

ROOT = Path(__file__).resolve().parents[2]


def test_readme_documents_only_full_evaluation_protocols():
    text = (ROOT / 'README.md').read_text()
    assert '50 episodes per task' in text
    assert '50 episodes per official task and three evaluation' in text
    assert 'NUM_EVAL=' not in text
    assert '10 episodes' not in text
    assert 'quick check' not in text.lower()


def test_release_has_only_paper_experiment_launchers():
    assert not list(ROOT.glob('*.sh'))
    script_paths = list((ROOT / 'experiments').rglob('*.sh'))
    scripts = {path.name for path in script_paths}
    assert scripts == {
        'eval_lewm_baseline_h100_4tasks.sh',
        'eval_lewm_baseline_h25_4tasks.sh',
        'eval_lewm_baseline_h50_4tasks.sh',
        'eval_lewm_baseline_h75_4tasks.sh',
        'eval_lewmpp_h100_4tasks.sh',
        'eval_lewmpp_h25_4tasks.sh',
        'eval_lewmpp_h50_4tasks.sh',
        'eval_lewmpp_h75_4tasks.sh',
        'precompute_lewm_latents_4tasks.sh',
        'train_action-prior-chunk_4tasks.sh',
        'train_lewm_4tasks.sh',
        'train_subgoal_latent_path_flow_h25_4tasks.sh',
        'train_subgoal_latent_path_flow_longh_4tasks.sh',
        'train_lewm_visual_ogbench8.sh',
        'precompute_visual_ogbench8_latents.sh',
        'train_action-prior-chunk_visual_ogbench8.sh',
        'train_latent_path_flow_visual_ogbench8.sh',
        'train_lewmpp_lerobot_v3.sh',
        'eval_lewmpp_visual_ogbench8.sh',
        'eval_lewm_baseline_visual_ogbench8.sh',
    }
    assert all(path.parent == ROOT / 'experiments' / 'train' for path in script_paths if path.name.startswith('train_'))
    for path in {path for path in script_paths if path.name.endswith('_4tasks.sh')}:
        text = path.read_text()
        assert '\nif ' not in text
        assert '\ncase ' not in text
        assert 'TASKS=(cube pusht reacher tworoom)' in text
        assert 'GPU_IDS=(0 1 2 3)' in text
        assert 'pids+=("$!")' in text
        assert 'wait "$pid" || status=1' in text
        assert 'exit "$status"' in text
    for retired in ('backup', 'reports', 'results'):
        assert not (ROOT / retired).exists()
    assert not list((ROOT / 'data_gen_scripts').rglob('*'))
    for retired in (
        'create_latent_subgoal_validation_manifest.py',
        'hyperparameters.sh',
        'main.py',
        'requirements.txt',
    ):
        assert not (ROOT / 'impls' / retired).exists()


def test_launchers_use_inline_path_configuration():
    scripts = [path for path in (ROOT / 'experiments').rglob('*.sh') if not path.name.startswith('_')]
    assert not list((ROOT / 'configs').glob('*.example.env'))
    for path in scripts:
        text = path.read_text()
        assert 'lewmpp_paths.env' not in text
        assert '# Fill in these paths before running this script.' in text
        assert '_ROOT=' in text
        for line in text.splitlines():
            if line.startswith(('LEWM_DATA_ROOT=', 'OGBENCH_DATA_ROOT=', 'LEWM_CHECKPOINT_ROOT=',
                                'ACTION_PRIOR_CHECKPOINT_ROOT=', 'LATENT_PATH_FLOW_CHECKPOINT_ROOT=')):
                assert line.endswith('=""')


def test_launchers_derive_generated_paths_from_common_roots():
    scripts = '\n'.join(path.read_text() for path in (ROOT / 'experiments').rglob('*.sh'))
    assert '$EXPERIMENT_ROOT/train/lewm/' in scripts
    assert '$EXPERIMENT_ROOT/train/action_prior/' in scripts
    assert '$EXPERIMENT_ROOT/train/latent_path_flow_h25/' in scripts
    assert '$EXPERIMENT_ROOT/train/latent_path_flow_longh/' in scripts
    assert '$EXPERIMENT_ROOT/eval/lewmpp_h25/' in scripts
    assert '$EXPERIMENT_ROOT/eval/lewm_h25/' in scripts
    assert '$EXPERIMENT_ROOT/train/lewm_latents/' in scripts
    assert '$EXPERIMENT_ROOT/train' in scripts
    assert '$EXPERIMENT_ROOT/eval/visual_ogbench_' in scripts


def test_action_prior_training_launcher_records_release_hyperparameters():
    text = (ROOT / 'experiments' / 'train' / 'train_action-prior-chunk_4tasks.sh').read_text()
    for argument in (
        '--train_steps=100000',
        '--save_interval=100000',
        '--log_interval=5000',
        '--batch_size=256',
        '--seed=777',
        '--lr=3e-4',
        '--discount=0.99',
        '--expectile=0.9',
        '--tau=0.005',
        '--chunk_size=5',
        '--alpha=3.0',
        '--p_aug=0.0',
        '--validation_fraction=0.05',
        '--representation_mode=all',
    ):
        assert argument in text


def test_lerobot_v3_launcher_runs_the_complete_rectangular_image_pipeline():
    text = (ROOT / 'experiments' / 'train' / 'train_lewmpp_lerobot_v3.sh').read_text()
    for value in (
        'yaoxianze/push_multi_red_cube',
        '5db612a1d18d4aa292fdbecb75a164c073ec0c82',
        'observation.images.camera_h',
        '--image-size=224',
        '--resize-mode=long_edge',
        'train_lewm_control.py',
        'precompute_lewm_lance_latents.py',
        'train_action_prior_chunk.py',
        'train_latent_path_flow_lewm_control.py',
    ):
        assert value in text


def test_paper_evaluations_explicitly_require_all_representation_sharing():
    for path in (ROOT / 'experiments' / 'eval').glob('eval_*.sh'):
        text = path.read_text()
        if '--action-prior-checkpoint-dir' in text:
            assert '--action-prior-representation-mode=all' in text


def test_control_suite_launchers_use_paper_cem_protocols():
    eval_root = ROOT / 'experiments' / 'eval'
    for path in eval_root.glob('eval_lewmpp_h*_4tasks.sh'):
        text = path.read_text()
        assert '--cem-iterations=5' in text
        assert '--cem-horizon=2' in text
        assert '--cem-receding-horizon=1' in text
        assert '--action-block=5' in text
    for path in eval_root.glob('eval_lewm_baseline_h*_4tasks.sh'):
        text = path.read_text()
        assert '--cem-iterations=30' in text
        assert '--cem-horizon=5' in text
        assert '--cem-receding-horizon=5' in text
        assert '--action-block=5' in text


def test_generator_launchers_record_and_validate_family_invariants():
    train_root = ROOT / 'experiments' / 'train'
    h25 = (train_root / 'train_subgoal_latent_path_flow_h25_4tasks.sh').read_text()
    assert '--goal-range=h25' in h25

    longh = (train_root / 'train_subgoal_latent_path_flow_longh_4tasks.sh').read_text()
    assert '--goal-range=full_future' in longh
    for text in (h25, longh):
        assert '--goal-sampling' not in text
        assert '--max-goal-steps' not in text
        assert '--hidden-dims' not in text
        assert '--model-dim=512' in text
    generator_evals = [
        path
        for path in (ROOT / 'experiments' / 'eval').glob('eval_*.sh')
        if '--subgoal-generator-checkpoint' in path.read_text()
    ]
    assert len(generator_evals) == 4
    for path in generator_evals:
        text = path.read_text()
        assert '--validate-only' in text


def test_python_entrypoints_match_the_release_pipeline():
    names = {path.name for path in (ROOT / 'impls').glob('*.py') if path.name.startswith(('train_', 'eval_'))}
    assert names == {
        'train_action_prior_chunk.py',
        'train_lewm_control.py',
        'train_latent_path_flow_lewm_control.py',
        'eval_lewm_control_suite.py',
        'train_lewm_ogbench.py',
        'train_action_prior_ogbench.py',
        'train_latent_path_flow_ogbench.py',
        'eval_visual_ogbench.py',
    }


def test_release_has_only_latent_path_flow_model_types():
    tree = ast.parse((ROOT / 'impls' / 'latent_path_flow_lewm_control.py').read_text())
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert classes == ['AdaLNTransformerBlock', 'LatentPathFlow']


def test_evaluator_exposes_required_release_variants_and_defaults():
    assert VARIANTS == ('full', 'no_subgoal', 'no_action_prior', 'no_moh', 'lewm', 'action_prior_chunk')
    assert (DEFAULT_CEM_ITERATIONS, DEFAULT_CEM_SAMPLES, DEFAULT_FLOW_STEPS) == (5, 300, 16)


def test_release_variants_keep_their_component_semantics():
    assert {variant: expected_components(variant) for variant in VARIANTS} == {
        'full': (True, True, 'moh', False),
        'no_subgoal': (False, True, 'moh', False),
        'no_action_prior': (True, False, 'moh', False),
        'no_moh': (True, True, 'last', False),
        'lewm': (False, False, 'last', False),
        'action_prior_chunk': (False, True, None, True),
    }


@pytest.mark.parametrize('variant', VARIANTS)
def test_every_release_variant_has_a_valid_runtime_configuration(variant):
    use_subgoal, use_prior, cost_mode, direct_policy = expected_components(variant)
    args = SimpleNamespace(
        variant=variant,
        action_prior_checkpoint_step=100_000,
        flow_sampling_steps=16,
        num_eval=1,
        goal_offset_steps=25,
        eval_budget=50,
        cem_horizon=2,
        cem_receding_horizon=1,
        action_block=5,
        cem_num_samples=300,
        cem_iterations=5,
        cem_topk=30,
        cem_var_scale=1.0,
        subgoal_generator_checkpoint='generator.msgpack' if use_subgoal else None,
        action_prior_checkpoint_dir='action_prior' if use_prior else None,
        action_prior_mode='policy_mode' if use_prior else 'zero',
        cem_cost_mode='moh' if direct_policy else cost_mode,
        generator_family='goalmax25' if use_subgoal else 'no_generator',
    )
    validate_args(args)


def test_full_variant_accepts_policy_mode_anchor():
    args = SimpleNamespace(
        variant='full',
        action_prior_checkpoint_step=100_000,
        flow_sampling_steps=16,
        num_eval=1,
        goal_offset_steps=25,
        eval_budget=50,
        cem_horizon=2,
        cem_receding_horizon=1,
        action_block=5,
        cem_num_samples=300,
        cem_iterations=5,
        cem_topk=30,
        cem_var_scale=1.0,
        subgoal_generator_checkpoint='generator.msgpack',
        action_prior_checkpoint_dir='action_prior',
        action_prior_mode='policy_mode_anchor',
        cem_cost_mode='moh',
        generator_family='goalmax25',
    )
    validate_args(args)


def test_action_prior_public_surface_uses_neutral_name():
    retired_method_name = 'gci' + 'ql'
    retired_loss_name = 'a' + 'wr'
    paths = [
        ROOT / 'impls' / 'train_action_prior_chunk.py',
        ROOT / 'impls' / 'action_prior_runtime_lewm_control.py',
        ROOT / 'impls' / 'agents' / 'action_prior_chunk.py',
        ROOT / 'impls' / 'eval_lewm_control_suite.py',
    ]
    paths.extend((ROOT / 'configs').glob('*.example.env'))
    paths.extend((ROOT / 'experiments').rglob('*.sh'))
    for path in paths:
        for line in path.read_text().splitlines():
            lowered = line.lower()
            assert retired_method_name not in lowered, path
            if 'actor_loss' not in lowered:
                assert retired_loss_name not in lowered, path


def test_preflight_rejects_generator_family_mismatch(tmp_path):
    (tmp_path / 'cube_single_expert.h5').touch()
    (tmp_path / 'cube_single_expert.lance').touch()
    lewm = tmp_path / 'lewm.msgpack'
    lewm.touch()
    checkpoint = tmp_path / 'checkpoint.msgpack'
    checkpoint.write_bytes(b'checkpoint')
    config = {
        'architecture': 'latent_path_flow_transformer_encoder',
        'task': 'cube',
        'latent_dataset': str(tmp_path / 'cube.h5'),
        'lewm_checkpoint_sha256': hashlib.sha256(lewm.read_bytes()).hexdigest(),
        'goal_sampling': 'uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25',
        'max_goal_steps': 25,
    }
    (tmp_path / 'config.json').write_text(json.dumps(config))
    args = SimpleNamespace(
        variant='no_action_prior',
        task='cube',
        data_root=str(tmp_path),
        lewm_checkpoint=str(lewm),
        action_prior_checkpoint_dir=None,
        action_prior_checkpoint_step=100_000,
        subgoal_generator_checkpoint=str(checkpoint),
        goal_offset_steps=25,
        generator_family='goalmax25',
        generator_type='latent_path_flow',
    )
    validate_release_files(args)
    args.generator_family = 'general_uniform_future'
    with pytest.raises(ValueError, match='H25 requires goalmax25'):
        validate_release_files(args)


def test_preflight_requires_the_requested_action_prior_representation(tmp_path):
    (tmp_path / 'cube_single_expert.h5').touch()
    (tmp_path / 'cube_single_expert.lance').touch()
    lewm = tmp_path / 'lewm.msgpack'
    lewm.touch()
    checkpoint_dir = tmp_path / 'cube_action_prior'
    checkpoint_dir.mkdir()
    (checkpoint_dir / 'params_100000.pkl').touch()
    flags = {
        'dataset_path': str(tmp_path / 'cube_single_expert.lance'),
        'seed': 777,
        'lewm_checkpoint_sha256': hashlib.sha256(lewm.read_bytes()).hexdigest(),
        'agent': {
            'chunk_size': 5,
            'representation_mode': 'v',
            'share_q_encoder': True,
            'share_v_encoder': True,
            'share_pi_encoder': False,
        },
        'representation': {'mode': 'v', 'q': 'lewm', 'v': 'lewm', 'pi': 'pixel'},
    }
    (checkpoint_dir / 'flags.json').write_text(json.dumps(flags))
    args = SimpleNamespace(
        variant='no_subgoal',
        task='cube',
        data_root=str(tmp_path),
        lewm_checkpoint=str(lewm),
        action_prior_checkpoint_dir=str(checkpoint_dir),
        action_prior_checkpoint_step=100_000,
        action_prior_representation_mode='v',
        action_block=5,
        subgoal_generator_checkpoint=None,
        goal_offset_steps=25,
        generator_family='no_generator',
        generator_type='latent_path_flow',
    )
    validate_release_files(args)
    args.action_prior_representation_mode = 'all'
    with pytest.raises(ValueError, match="uses representation mode 'v'"):
        validate_release_files(args)
