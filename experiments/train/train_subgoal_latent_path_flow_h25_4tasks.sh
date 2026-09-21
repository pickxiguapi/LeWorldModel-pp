#!/usr/bin/env bash
set -euo pipefail

# Fill in these paths before running this script.
EXPERIMENT_ROOT="outputs"

TASKS=(cube pusht reacher tworoom)
GPU_IDS=(0 1 2 3)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

pids=()
for index in "${!TASKS[@]}"; do
  task=${TASKS[$index]}
  save_dir="$EXPERIMENT_ROOT/train/latent_path_flow_h25/$task"
  mkdir -p "$save_dir"
  (
    export CUDA_VISIBLE_DEVICES=${GPU_IDS[$index]}
    python impls/train_latent_path_flow_lewm_control.py \
      --latent-dataset="$EXPERIMENT_ROOT/train/lewm_latents/$task.h5" \
      --save-dir="$save_dir" \
      --exp-name="lewmpp_goalmax25_latent_path_flow_${task}_seed0" \
      --goal-range=h25 \
      --seed=0 \
      --split-seed=0 \
      --train-fraction=0.95 \
      --subgoal-steps=10 \
      --action-block=5 \
      --history-size=3 \
      --train-steps=200000 \
      --batch-size=1024 \
      --model-dim=512 \
      --depth=4 \
      --num-heads=8 \
      --ff-dim=2048 \
      --time-dim=64 \
      --flow-sampling-steps=16 \
      --ema-decay=0.9999 \
      --learning-rate=1e-4 \
      --final-learning-rate=1e-5 \
      --warmup-steps=5000 \
      --weight-decay=1e-4 \
      --gradient-clip=1.0 \
      --validation-pairs=10000 \
      --eval-batch-size=1024 \
      --log-interval=1000 \
      --eval-interval=10000 \
      --checkpoint-interval=25000 \
      --resume \
      2>&1 | tee "$save_dir/train.log"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
