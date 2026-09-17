#!/usr/bin/env bash
set -euo pipefail

# Fill in these paths before running this script.
LEWM_DATA_ROOT=""
LEWM_CHECKPOINT_ROOT=""
EXPERIMENT_ROOT="outputs"

TASKS=(cube pusht reacher tworoom)
GPU_IDS=(0 1 2 3)
DATASETS=(
  "$LEWM_DATA_ROOT/cube_single_expert.lance"
  "$LEWM_DATA_ROOT/pusht_expert_train.lance"
  "$LEWM_DATA_ROOT/reacher.lance"
  "$LEWM_DATA_ROOT/tworoom.lance"
)
LEWM_CHECKPOINTS=(
  "$LEWM_CHECKPOINT_ROOT/cube/weights_epoch_10.msgpack"
  "$LEWM_CHECKPOINT_ROOT/pusht/weights_epoch_10.msgpack"
  "$LEWM_CHECKPOINT_ROOT/reacher/weights_epoch_10.msgpack"
  "$LEWM_CHECKPOINT_ROOT/tworoom/weights_epoch_10.msgpack"
)

export XLA_PYTHON_CLIENT_PREALLOCATE=false

pids=()
for index in "${!TASKS[@]}"; do
  task=${TASKS[$index]}
  output="$EXPERIMENT_ROOT/train/lewm_latents/$task.h5"
  log_file="$EXPERIMENT_ROOT/train/lewm_latents/$task.log"
  mkdir -p "$EXPERIMENT_ROOT/train/lewm_latents"

  (
    export CUDA_VISIBLE_DEVICES=${GPU_IDS[$index]}
    python impls/precompute_lewm_lance_latents.py \
      --task="$task" \
      --lance-path="${DATASETS[$index]}" \
      --checkpoint="${LEWM_CHECKPOINTS[$index]}" \
      --output="$output" \
      --batch-size=512 \
      --decode-workers=12 \
      --output-dtype=float32 \
      --flush-every-batches=20 \
      --log-every-batches=20 \
      2>&1 | tee "$log_file"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
