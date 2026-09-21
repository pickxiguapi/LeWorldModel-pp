#!/usr/bin/env bash
set -euo pipefail

# Fill in these paths before running this script.
OGBENCH_DATA_ROOT=""
EXPERIMENT_ROOT="outputs"
LEWM_CHECKPOINT_ROOT=""
EVAL_ROOT="$EXPERIMENT_ROOT/eval/visual_ogbench_lewm"
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
read -r -a GPU_IDS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a EVAL_SEEDS <<< "${EVAL_SEEDS:-0 1 42}"
NUM_EVAL=${NUM_EVAL:-50}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless

eval_one() {
  local gpu=$1 index=$2 env_name=${ENVS[$2]} tag=${TAGS[$2]}
  local output_dir="$EVAL_ROOT/$tag/seed$CURRENT_EVAL_SEED"
  local output="$output_dir/result.json"
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python impls/eval_visual_ogbench.py \
      --env-name="$env_name" --dataset-path="$OGBENCH_DATA_ROOT/$env_name.npz" \
      --policy-guidance=none \
      --lewm-checkpoint="$LEWM_CHECKPOINT_ROOT/$tag/weights_epoch_10.msgpack" \
      --num-eval="$NUM_EVAL" --seed="$CURRENT_EVAL_SEED" --cem-horizon=5 \
      --cem-receding-horizon=1 --action-block=5 --cem-num-samples=300 \
      --cem-iterations=30 --cem-topk=30 --cem-var-scale=1.0 \
      --cem-cost-mode=moh --output="$output" 2>&1 | tee "$output_dir/eval.log"
}

for CURRENT_EVAL_SEED in "${EVAL_SEEDS[@]}"; do
  export CURRENT_EVAL_SEED
  pids=()
  for index in "${!ENVS[@]}"; do
    eval_one "${GPU_IDS[$index]}" "$index" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  (( status == 0 )) || exit "$status"
done

if [[ "$NUM_EVAL" == 50 && "${EVAL_SEEDS[*]}" == "0 1 42" ]]; then
  python impls/aggregate_visual_ogbench_results.py \
    --method=lewm --results-root="$EVAL_ROOT" --output="$EVAL_ROOT/summary.json"
fi
