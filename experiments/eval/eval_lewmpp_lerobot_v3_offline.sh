#!/usr/bin/env bash
set -euo pipefail

# Fill in these paths before running this script.
EXPERIMENT_ROOT="outputs"

DATASET_TAG="push_multi_red_cube"
GPU_ID=0

LANCE_DATASET="$EXPERIMENT_ROOT/data/$DATASET_TAG/$DATASET_TAG.lance"
TRAIN_ROOT="$EXPERIMENT_ROOT/train/lerobot_v3/$DATASET_TAG"
LEWM_CHECKPOINT="$TRAIN_ROOT/lewm/weights_epoch_50.msgpack"
ACTION_PRIOR_DIR="$TRAIN_ROOT/action_prior"
LATENT_DATASET="$TRAIN_ROOT/lewm_latents.h5"
LATENT_PATH_FLOW_CHECKPOINT="$TRAIN_ROOT/latent_path_flow/checkpoint_100000.msgpack"
RESULT_DIR="$EXPERIMENT_ROOT/eval/lerobot_v3/$DATASET_TAG"

mkdir -p "$RESULT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python impls/eval_lerobot_v3_offline.py \
  --lance-path="$LANCE_DATASET" \
  --lewm-checkpoint="$LEWM_CHECKPOINT" \
  --action-prior-dir="$ACTION_PRIOR_DIR" \
  --action-prior-step=100000 \
  --latent-dataset="$LATENT_DATASET" \
  --latent-path-flow-checkpoint="$LATENT_PATH_FLOW_CHECKPOINT" \
  --output="$RESULT_DIR/offline_metrics.json" \
  --train-fraction=0.96 \
  --split-seed=0 \
  --eval-seed=0 \
  --batch-size=128 \
  --action-prior-samples=10000 \
  --flow-validation-pairs=10000 \
  --flow-batch-size=1024 \
  2>&1 | tee "$RESULT_DIR/eval.log"
