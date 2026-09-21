# LeWorldModel++

**LeWorldModel++ (LeWM++)** is a planning-interface framework for long-horizon image-goal control with frozen latent world models (e.g. LeWorldModel, DINO-WM). It addresses three coupled failures that arise when short model rollouts must serve distant goals: mismatched planning targets, uninformed action search, and restrictive terminal-state costs.

LeWM++ combines **LatentPathFlow** to generate reachable local targets, an **Action Chunk Prior** to initialize CEM from goal-directed offline behavior, and **Min-over-Horizon (MoH)** to score the closest predicted approach to the target. The controller executes one optimized action chunk and replans, preserving global intent through the final image goal while operating within a locally reliable prediction horizon.

## Installation

```bash
git clone https://github.com/pickxiguapi/LeWorldModel-pp.git
cd LeWorldModel-pp
uv sync --extra train --extra dev
```

## Training

Run all commands in this README from the repository root. Activate the
environment first:

```bash
source .venv/bin/activate
```

All launchers write checkpoints, latent caches, logs, and evaluation results
to `outputs/` by default. Change `EXPERIMENT_ROOT` at the top of a launcher if
you want a different location. Dataset and external-checkpoint paths remain
empty and must be filled in explicitly.

### LeWM Control Suite

1. Download the four official HDF5 datasets from the
[LeWM Hugging Face collection](https://huggingface.co/collections/quentinll/lewm):

- `quentinll/lewm-cube`: `cube_single_expert.tar.zst`;
- `quentinll/lewm-pusht`: `pusht_expert_train.h5.zst`;
- `quentinll/lewm-reacher`: `reacher.tar.zst`;
- `quentinll/lewm-tworooms`: `tworoom.tar.zst`.

2. Extract the archives and place the four HDF5 files in one directory using the
filenames below. Then convert each HDF5 file to its JPEG-backed Lance table:

```bash
LEWM_DATA_ROOT=/absolute/path/to/lewm-control-suite

for name in cube_single_expert pusht_expert_train reacher tworoom; do
  python scripts/convert_lewm_hdf5_to_lance.py \
    "$LEWM_DATA_ROOT/$name.h5" \
    "$LEWM_DATA_ROOT/$name.lance"
done
```

The prepared dataset directory should have this layout:

```text
lewm-control-suite/
├── cube_single_expert.h5
├── cube_single_expert.lance/
├── pusht_expert_train.h5
├── pusht_expert_train.lance/
├── reacher.h5
├── reacher.lance/
├── tworoom.h5
└── tworoom.lance/
```

After setting the paths at the top of each launcher, train all components in
dependency order:

```bash
# 3. Train the LeWM dynamics models.
bash experiments/train/train_lewm_4tasks.sh

# 4. Encode the offline datasets with the trained LeWM checkpoints.
bash experiments/precompute_lewm_latents_4tasks.sh

# 5. Train the final-goal-conditioned Action Chunk Priors.
bash experiments/train/train_action-prior-chunk_4tasks.sh

# 6. Train the H25 LatentPathFlow models.
bash experiments/train/train_subgoal_latent_path_flow_h25_4tasks.sh

# 7. Train the long-horizon LatentPathFlow models used for H50/H75/H100.
bash experiments/train/train_subgoal_latent_path_flow_longh_4tasks.sh
```

### Visual OGBench

Expected filenames under `OGBENCH_DATA_ROOT` are the eight environment names ending in `.npz`, plus matching `-val.npz` files, such as `visual-cube-single-play-v0.npz` and `visual-cube-single-play-v0-val.npz`.

```bash
# 1. Train one frozen LeWM model for each Visual OGBench dataset.
bash experiments/train/train_lewm_visual_ogbench8.sh

# 2. Precompute checkpoint-bound latent datasets.
bash experiments/precompute_visual_ogbench8_latents.sh

# 3. Train the final-goal-conditioned Action Chunk Priors.
bash experiments/train/train_action-prior-chunk_visual_ogbench8.sh

# 4. Train the general-uniform-future LatentPathFlow models.
bash experiments/train/train_latent_path_flow_visual_ogbench8.sh
```

Visual OGBench training artifacts are written below `EXPERIMENT_ROOT/train/`.
Set the evaluation launchers' checkpoint-root variables to the corresponding
training directories when evaluating your own models.

### LeRobot v3 real-robot datasets

Install the LeRobot dataset adapter in the same environment:

```bash
uv sync --extra train --extra robot --extra dev
source .venv/bin/activate
```

The example launcher trains LeWM++ from the LeRobotDataset v3.0 dataset
[`yaoxianze/push_multi_red_cube`](https://huggingface.co/datasets/yaoxianze/push_multi_red_cube).
Open `experiments/train/train_lewmpp_lerobot_v3.sh` and fill in
`LEROBOT_DATA_ROOT` with a local directory for the downloaded LeRobot data.
Then run:

```bash
bash experiments/train/train_lewmpp_lerobot_v3.sh
```

The launcher pins the source dataset revision and uses
`observation.images.camera_h`. Its 480 x 640 RGB frames are resized without
cropping or distortion: the aspect ratio is preserved and the long edge is
set to 224, producing 168 x 224 training frames. It then runs the complete
offline training dependency chain:

1. convert the selected LeRobot camera and 14-dimensional actions to a
   JPEG-backed Lance trajectory dataset;
2. train the frozen LeWM encoder and dynamics model;
3. precompute checkpoint-bound LeWM latents;
4. train the Action Chunk Prior and full-future LatentPathFlow.

After `lewm_latents.h5` exists, LatentPathFlow can run independently on a
second GPU while the Action Chunk Prior is still training.

```bash
bash experiments/train/train_latent_path_flow_lerobot_v3.sh
```

Converted data is written to `outputs/data/push_multi_red_cube/`. Checkpoints,
the latent cache, and logs are written below
`outputs/train/lerobot_v3/push_multi_red_cube/`. Change `EXPERIMENT_ROOT` in
both launchers to place all generated artifacts elsewhere. The final files
consumed by evaluation are:

```text
outputs/
├── data/push_multi_red_cube/push_multi_red_cube.lance/
└── train/lerobot_v3/push_multi_red_cube/
    ├── lewm/weights_epoch_50.msgpack
    ├── lewm_latents.h5
    ├── action_prior/params_100000.pkl
    └── latent_path_flow/checkpoint_50000.msgpack
```

This is the offline real-robot-data training pipeline. The real-robot API below
loads the Lance dataset only to restore the training action normalization and
bounds; the latent cache is not needed for deployment.

### Real-robot inference

Real-robot deployment requires a thin adapter between the LeWM++ policy and the
robot stack. The adapter should reproduce the observation preprocessing used
during training, load the corresponding model artifacts and action statistics,
and translate policy outputs into the robot's native command representation.
The reference implementation is `impls/real_robot_lewmpp.py`.

Construct the policy from paths supplied by your deployment configuration
rather than embedding machine-specific directories in the robot program:

```python
from real_robot_lewmpp import RealRobotLeWMPPPolicy

policy = RealRobotLeWMPPPolicy(
    lance_path=config.dataset_path,
    lewm_checkpoint=config.lewm_checkpoint,
    action_prior_dir=config.action_prior_dir,
    action_prior_step=config.action_prior_step,
    latent_path_flow_checkpoint=config.latent_path_flow_checkpoint,
    input_color=config.camera_color_format,
)
```

At the beginning of an episode, provide the desired-goal image and optionally
compile the planning path before enabling motion. During execution, pass each
new observation to the streaming interface and apply the returned action. The
policy retains the required observation history and replans at the configured
action-chunk boundary.

```python
initial_frame = robot.get_observation()
policy.warmup(initial_frame, goal_image)
policy.reset(goal_image)

while not robot.is_done():
    frame = robot.get_observation()
    action = policy.act(frame)
    robot.send_action(action)
```

The following parts must match the data and configuration used for training:

- camera selection, color convention, image geometry, and temporal sampling;
- action dimension, column order, units, normalization, and command frequency;
- LeWM, Action Chunk Prior, and LatentPathFlow checkpoints from the same run;
- planning horizon, chunk length, and any platform-specific action limits.

The bundled adapter is a reference for the example dataset and should be
adjusted at its platform boundary for a different observation or action schema.
`warmup` performs JAX compilation without sending its planned action to the
robot. Independent of the policy, the robot-side program remains responsible
for collision checking, emergency stops, workspace and velocity limits,
watchdogs, and all other hardware safety interlocks.

## Pretrained artifacts

The exact checkpoints selected for the release evaluation are stored in
[`IffYuan/LeWorldModelplusplus`](https://huggingface.co/IffYuan/LeWorldModelplusplus).
Download them with:

```bash
uvx --from huggingface_hub hf download IffYuan/LeWorldModelplusplus \
  --include "*/checkpoints/**" --local-dir artifacts
```

The checkpoint repository uses this layout:

```text
artifacts/
├── lewm-control-suite/
│   └── checkpoints/
│       ├── lewm/{cube,pusht,reacher,tworoom}/
│       ├── action-prior/{cube,pusht,reacher,tworoom}/
│       └── latent-path-flow/{h25,longh}/{cube,pusht,reacher,tworoom}/
└── visual-ogbench/
    └── checkpoints/{lewm,action-prior,latent-path-flow}/<dataset-tag>/
```

Datasets are not included. Prepare them as described in Training above.
Keep the downloaded `config.json` and `flags.json` files beside their weights;
they are required to restore the models.

LeWM weights are named `weights_epoch_10.msgpack`, and LatentPathFlow weights
are named `checkpoint_200000.msgpack`. Action Prior weights are
`params_100000.pkl` for the Control Suite and `params_500000.pkl` for Visual OGBench.


## Evaluation

Activate the installed environment and run all commands from the repository root:

```bash
source .venv/bin/activate
```

Edit the path assignments **inside each bash launcher** before running it.
The examples below assume checkpoints were downloaded to `artifacts/` in the
repository root. Set dataset paths to your local data directories and choose
`GPU_IDS` for your machine. Evaluation with pretrained checkpoints does not
require training or latent-cache precomputation.

### Planning effectiveness and long-horizon scaling (LeWM Control Suite)

In each `eval_lewmpp_h{25,50,75,100}_4tasks.sh`, set:

```bash
LEWM_DATA_ROOT="/absolute/path/to/lewm-control-suite"
EXPERIMENT_ROOT="outputs"
LEWM_CHECKPOINT_ROOT="artifacts/lewm-control-suite/checkpoints/lewm"
ACTION_PRIOR_CHECKPOINT_ROOT="artifacts/lewm-control-suite/checkpoints/action-prior"
```

Set `LATENT_PATH_FLOW_CHECKPOINT_ROOT` according to the launcher:

| Launcher | `LATENT_PATH_FLOW_CHECKPOINT_ROOT` |
| --- | --- |
| `eval_lewmpp_h25_4tasks.sh` | `artifacts/lewm-control-suite/checkpoints/latent-path-flow/h25` |
| `eval_lewmpp_h50_4tasks.sh`, `eval_lewmpp_h75_4tasks.sh`, `eval_lewmpp_h100_4tasks.sh` | `artifacts/lewm-control-suite/checkpoints/latent-path-flow/longh` |

These roots contain the four task directories; do not append a task name or
checkpoint filename. `LEWM_DATA_ROOT` must contain the four HDF5 files and their
converted Lance tables shown in Training above. In each
`eval_lewm_baseline_h{25,50,75,100}_4tasks.sh`, fill in only `LEWM_DATA_ROOT`,
`EXPERIMENT_ROOT`, and `LEWM_CHECKPOINT_ROOT`; the baseline requires only the HDF5 data.

The LeWM baseline follows the official CEM300x30, H5/RH5 protocol with an
action block of 5. LeWM++ uses CEM300x5 and H2/RH1 with the same action block.

Run all four evaluation horizons (50 episodes per task):

```bash
for horizon in 25 50 75 100; do
  bash "experiments/eval/eval_lewmpp_h${horizon}_4tasks.sh"
  bash "experiments/eval/eval_lewm_baseline_h${horizon}_4tasks.sh"
done

python impls/aggregate_lewm_control_results.py \
  --results-root outputs/eval \
  --output outputs/eval/lewm_control_suite_summary.csv
```

Results are written to `outputs/eval/{lewmpp,lewm}_h<horizon>/<task>/seed<seed>/`.
If you change `EXPERIMENT_ROOT`, adjust the aggregation paths accordingly.

### More challenging tasks on Visual OGBench

In `eval_lewmpp_visual_ogbench8.sh`, set:

```bash
OGBENCH_DATA_ROOT="/absolute/path/to/visual-ogbench-data"
EXPERIMENT_ROOT="outputs"
LEWM_CHECKPOINT_ROOT="artifacts/visual-ogbench/checkpoints/lewm"
ACTION_PRIOR_CHECKPOINT_ROOT="artifacts/visual-ogbench/checkpoints/action-prior"
LATENT_PATH_FLOW_CHECKPOINT_ROOT="artifacts/visual-ogbench/checkpoints/latent-path-flow"
```

The checkpoint roots contain `cs_play`, `cd_play`, `ct_play`, `scene_play`,
`cs_noisy`, `cd_noisy`, `ct_noisy`, and `scene_noisy`. `OGBENCH_DATA_ROOT` must
contain the eight training NPZ files named after the environments, such as
`visual-cube-single-play-v0.npz`; these are used for action normalization.
In `eval_lewm_baseline_visual_ogbench8.sh`, set only `OGBENCH_DATA_ROOT`,
`EXPERIMENT_ROOT`, and `LEWM_CHECKPOINT_ROOT`.

Run all eight datasets with 50 episodes per official task and three evaluation
seeds. Both launchers write an aggregate summary automatically:

```bash
bash experiments/eval/eval_lewmpp_visual_ogbench8.sh
bash experiments/eval/eval_lewm_baseline_visual_ogbench8.sh
```

To regenerate the LeWM++ summary from completed results without rerunning the
environments (the full 50-episode, three-seed evaluation):

```bash
python impls/aggregate_visual_ogbench_results.py \
  --results-root outputs/eval/visual_ogbench_lewmpp \
  --output outputs/eval/visual_ogbench_lewmpp/summary.json
```

## Acknowledgments

LeWM++ builds on [LeWorldModel](https://github.com/lucas-maes/le-wm) and [OGBench](https://github.com/seohongpark/ogbench). We thank their authors for releasing the latent world-model implementation, benchmark environments, datasets, and evaluation APIs that make this work possible. The DINO-WM transfer experiments described in the paper use the original [DINO-WM](https://github.com/gaoyuezhou/dino_wm) and are not included in this repository.

## License and citation

The repository retains the MIT license. A LeWM++ citation block will be added when the paper receives a public identifier.
