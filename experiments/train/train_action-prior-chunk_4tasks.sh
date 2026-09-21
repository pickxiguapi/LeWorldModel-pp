#!/usr/bin/env bash
set -euo pipefail

# Fill in these paths before running this script.
LEWM_DATA_ROOT=""
EXPERIMENT_ROOT="outputs"
LEWM_CHECKPOINT_ROOT=""

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
  save_dir="$EXPERIMENT_ROOT/train/action_prior/$task"
  log_file="$save_dir/train.log"
  mkdir -p "$save_dir"

  (
    export CUDA_VISIBLE_DEVICES=${GPU_IDS[$index]}
    python impls/train_action_prior_chunk.py \
      --dataset_path="${DATASETS[$index]}" \
      --lewm_checkpoint="${LEWM_CHECKPOINTS[$index]}" \
      --save_dir="$save_dir" \
      --train_steps=100000 \
      --save_interval=100000 \
      --log_interval=5000 \
      --batch_size=256 \
      --seed=777 \
      --lr=3e-4 \
      --discount=0.99 \
      --expectile=0.9 \
      --tau=0.005 \
      --chunk_size=5 \
      --alpha=3.0 \
      --p_aug=0.0 \
      --validation_fraction=0.05 \
      --representation_mode=all \
      2>&1 | tee "$log_file"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
