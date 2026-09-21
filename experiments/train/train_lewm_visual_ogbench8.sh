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
  local save_dir="$EXPERIMENT_ROOT/train/visual_ogbench_lewm/$tag"
  mkdir -p "$save_dir"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python impls/train_lewm_ogbench.py \
      --dataset_path="$OGBENCH_DATA_ROOT/$env_name.npz" \
      --validation_dataset_path="$OGBENCH_DATA_ROOT/$env_name-val.npz" \
      --dataset_format=npz --save_dir="$save_dir" --exp_name="lewm_ogbench8_$tag" \
      --epochs=10 --batch_size=128 --seed=3072 --frameskip=5 --image_size=64 \
      --learning_rate=5e-5 --weight_decay=1e-3 --sigreg_weight=0.09 \
      --sigreg_knots=17 --sigreg_num_proj=1024 --decode_workers=1 \
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
