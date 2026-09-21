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
  local gpu=$1 index=$2 tag=${TAGS[$2]}
  local save_dir="$EXPERIMENT_ROOT/train/visual_ogbench_latent_path_flow/$tag"
  mkdir -p "$save_dir"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python impls/train_latent_path_flow_ogbench.py \
      --latent-dataset="$EXPERIMENT_ROOT/train/visual_ogbench_latents/$tag.h5" --save-dir="$save_dir" \
      --exp-name="latent_path_flow_ogbench8_$tag" \
      --history-size=3 --hidden-dim=512 --depth=4 --num-heads=8 --ff-dim=2048 \
      --time-dim=64 --seed=0 --split-seed=0 --train-fraction=0.95 \
      --subgoal-steps=10 --action-block=5 --goal-sampling=uniform_future \
      --train-steps=200000 --batch-size=1024 --flow-sampling-steps=16 \
      --ema-decay=0.9999 \
      --learning-rate=1e-4 --final-learning-rate=1e-5 --warmup-steps=5000 \
      --weight-decay=1e-4 --gradient-clip=1.0 --validation-pairs=10000 \
      --eval-batch-size=1024 --log-interval=1000 --eval-interval=10000 \
      --checkpoint-interval=25000 --resume 2>&1 | tee "$save_dir/train.log"
}

pids=()
for index in "${!TAGS[@]}"; do
  train_one "${GPU_IDS[$index]}" "$index" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
