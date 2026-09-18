#!/usr/bin/env bash
set -euo pipefail

# Fill in these paths before running this script.
EXPERIMENT_ROOT="outputs"

DATASET_TAG="push_multi_red_cube"
GPU_ID=0

LATENT_DATASET="$EXPERIMENT_ROOT/train/lerobot_v3/$DATASET_TAG/lewm_latents.h5"
LATENT_PATH_FLOW_DIR="$EXPERIMENT_ROOT/train/lerobot_v3/$DATASET_TAG/latent_path_flow"

mkdir -p "$LATENT_PATH_FLOW_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python impls/train_latent_path_flow_lewm_control.py \
  --latent-dataset="$LATENT_DATASET" \
  --save-dir="$LATENT_PATH_FLOW_DIR" \
  --exp-name="lewmpp_lerobot_v3_${DATASET_TAG}_seed0" \
  --goal-range=full_future \
  --seed=0 \
  --split-seed=0 \
  --train-fraction=0.96 \
  --subgoal-steps=20 \
  --action-block=10 \
  --history-size=3 \
  --train-steps=50000 \
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
  --eval-interval=5000 \
  --checkpoint-interval=10000 \
  --resume \
  2>&1 | tee "$LATENT_PATH_FLOW_DIR/train.log"
