#!/usr/bin/env bash
set -euo pipefail

# Fill in these paths before running this script.
LEWM_DATA_ROOT=""
EXPERIMENT_ROOT="outputs"

TASKS=(cube pusht reacher tworoom)
GPU_IDS=(0 1 2 3)
SEEDS=(0 666 3072 1)
DATASETS=(
  "$LEWM_DATA_ROOT/cube_single_expert.lance"
  "$LEWM_DATA_ROOT/pusht_expert_train.lance"
  "$LEWM_DATA_ROOT/reacher.lance"
  "$LEWM_DATA_ROOT/tworoom.lance"
)

export XLA_PYTHON_CLIENT_PREALLOCATE=false

pids=()
for index in "${!TASKS[@]}"; do
  task=${TASKS[$index]}
  seed=${SEEDS[$index]}
  save_dir="$EXPERIMENT_ROOT/train/lewm/$task"
  log_file="$save_dir/train.log"
  mkdir -p "$save_dir"

  (
    export CUDA_VISIBLE_DEVICES=${GPU_IDS[$index]}
    python impls/train_lewm_control.py \
      --dataset_path="${DATASETS[$index]}" \
      --save_dir="$save_dir" \
      --exp_name="lewm_${task}_seed${seed}" \
      --decode_workers=6 \
      --seed="$seed" \
      --epochs=10 \
      --batch_size=128 \
      --frameskip=5 \
      --image_size=224 \
      --learning_rate=5e-5 \
      --weight_decay=1e-3 \
      --sigreg_weight=0.09 \
      --sigreg_knots=17 \
      --sigreg_num_proj=1024 \
      2>&1 | tee "$log_file"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
