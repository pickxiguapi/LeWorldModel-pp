#!/usr/bin/env bash
set -euo pipefail

# Fill in these paths before running this script.
LEROBOT_DATA_ROOT=""
EXPERIMENT_ROOT="outputs"

DATASET_REPO_ID="yaoxianze/push_multi_red_cube"
DATASET_REVISION="5db612a1d18d4aa292fdbecb75a164c073ec0c82"
DATASET_TAG="push_multi_red_cube"
CAMERA_KEY="observation.images.camera_h"
GPU_ID=0

DATASET_DIR="$EXPERIMENT_ROOT/data/$DATASET_TAG"
LANCE_DATASET="$DATASET_DIR/$DATASET_TAG.lance"
LEWM_DIR="$EXPERIMENT_ROOT/train/lerobot_v3/$DATASET_TAG/lewm"
LATENT_DATASET="$EXPERIMENT_ROOT/train/lerobot_v3/$DATASET_TAG/lewm_latents.h5"
ACTION_PRIOR_DIR="$EXPERIMENT_ROOT/train/lerobot_v3/$DATASET_TAG/action_prior"
LATENT_PATH_FLOW_DIR="$EXPERIMENT_ROOT/train/lerobot_v3/$DATASET_TAG/latent_path_flow"

mkdir -p "$DATASET_DIR" "$LEWM_DIR" "$ACTION_PRIOR_DIR" "$LATENT_PATH_FLOW_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python scripts/convert_lerobot_v3_to_lance.py \
  --repo-id="$DATASET_REPO_ID" \
  --revision="$DATASET_REVISION" \
  --root="$LEROBOT_DATA_ROOT" \
  --camera-key="$CAMERA_KEY" \
  --destination="$LANCE_DATASET" \
  --image-size=224 \
  --resize-mode=long_edge \
  --jpeg-quality=95 \
  --batch-size=64 \
  --decode-workers=4 \
  --encode-workers=8 \
  2>&1 | tee "$DATASET_DIR/convert.log"

python impls/train_lewm_control.py \
  --dataset_path="$LANCE_DATASET" \
  --save_dir="$LEWM_DIR" \
  --exp_name="lewm_${DATASET_TAG}_seed3072" \
  --decode_workers=6 \
  --seed=3072 \
  --epochs=50 \
  --save_interval_epochs=10 \
  --batch_size=128 \
  --frameskip=10 \
  --train_fraction=0.96 \
  --episode_split \
  --split_seed=0 \
  --image_size=224 \
  --learning_rate=5e-5 \
  --weight_decay=1e-3 \
  --sigreg_weight=0.09 \
  --sigreg_knots=17 \
  --sigreg_num_proj=1024 \
  2>&1 | tee "$LEWM_DIR/train.log"

python impls/precompute_lewm_lance_latents.py \
  --task="$DATASET_TAG" \
  --lance-path="$LANCE_DATASET" \
  --checkpoint="$LEWM_DIR/weights_epoch_50.msgpack" \
  --output="$LATENT_DATASET" \
  --batch-size=512 \
  --decode-workers=12 \
  --output-dtype=float32 \
  --flush-every-batches=20 \
  --log-every-batches=20 \
  2>&1 | tee "$EXPERIMENT_ROOT/train/lerobot_v3/$DATASET_TAG/precompute_latents.log"

python impls/train_action_prior_chunk.py \
  --dataset_path="$LANCE_DATASET" \
  --lewm_checkpoint="$LEWM_DIR/weights_epoch_50.msgpack" \
  --save_dir="$ACTION_PRIOR_DIR" \
  --train_steps=100000 \
  --save_interval=25000 \
  --log_interval=5000 \
  --batch_size=256 \
  --seed=777 \
  --lr=3e-4 \
  --discount=0.99 \
  --expectile=0.9 \
  --tau=0.005 \
  --chunk_size=10 \
  --alpha=3.0 \
  --p_aug=0.0 \
  --validation_fraction=0.04 \
  --episode_split_seed=0 \
  --representation_mode=all \
  2>&1 | tee "$ACTION_PRIOR_DIR/train.log"

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
  --train-steps=100000 \
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
  2>&1 | tee "$LATENT_PATH_FLOW_DIR/train.log"
