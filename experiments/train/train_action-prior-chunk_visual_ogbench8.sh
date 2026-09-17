#!/usr/bin/env bash
set -euo pipefail

# Fill in these paths before running this script.
OGBENCH_DATA_ROOT=""
EXPERIMENT_ROOT="outputs"
ENVS=(
  visual-cube-single-play-v0
  visual-cube-double-play-v0
  visual-cube-triple-play-v0
  visual-scene-play-v0
  visual-cube-single-noisy-v0
  visual-cube-double-noisy-v0
  visual-cube-triple-noisy-v0
  visual-scene-noisy-v0
)
TAGS=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)
GPU_IDS=(0 1 2 3 4 5 6 7)

train_one() {
  local gpu=$1 index=$2 env_name=${ENVS[$2]} tag=${TAGS[$2]}
  local save_dir="$EXPERIMENT_ROOT/train/visual_ogbench_action_prior/$tag"
  mkdir -p "$save_dir"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python impls/train_action_prior_ogbench.py \
      --env_name="$env_name" --dataset_path="$OGBENCH_DATA_ROOT/$env_name.npz" \
      --save_dir="$save_dir" --representation_mode=independent \
      --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug=0.5 \
      --train_steps=500000 --batch_size=512 --seed=0 --chunk_size=5 \
      --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
      --log_interval=5000 --save_interval=100000 \
      2>&1 | tee "$save_dir/train.log"
}

pids=()
for index in "${!ENVS[@]}"; do
  train_one "${GPU_IDS[$index]}" "$index" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
