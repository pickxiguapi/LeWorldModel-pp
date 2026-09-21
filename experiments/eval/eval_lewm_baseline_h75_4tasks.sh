#!/usr/bin/env bash
set -euo pipefail

# Fill in these paths before running this script.
LEWM_DATA_ROOT=""
EXPERIMENT_ROOT="outputs"
LEWM_CHECKPOINT_ROOT=""

TASKS=(cube pusht reacher tworoom)
GPU_IDS=(0 1 2 3)
EVAL_SEEDS=(0 1 42)
LEWM_CHECKPOINTS=(
  "$LEWM_CHECKPOINT_ROOT/cube/weights_epoch_10.msgpack"
  "$LEWM_CHECKPOINT_ROOT/pusht/weights_epoch_10.msgpack"
  "$LEWM_CHECKPOINT_ROOT/reacher/weights_epoch_10.msgpack"
  "$LEWM_CHECKPOINT_ROOT/tworoom/weights_epoch_10.msgpack"
)

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless

pids=()
for index in "${!TASKS[@]}"; do
  task=${TASKS[$index]}
  (
    export CUDA_VISIBLE_DEVICES=${GPU_IDS[$index]}
    for seed in "${EVAL_SEEDS[@]}"; do
      result_dir="$EXPERIMENT_ROOT/eval/lewm_h75/$task/seed${seed}"
      mkdir -p "$result_dir"
      args=(
        --task="$task"
        --variant=lewm
        --experiment-group=lewm_h75
        --generator-family=no_generator
        --data-root="$LEWM_DATA_ROOT"
        --lewm-checkpoint="${LEWM_CHECKPOINTS[$index]}"
        --action-prior-mode=zero
        --num-eval=50
        --seed="$seed"
        --goal-offset-steps=75
        --eval-budget=150
        --cem-horizon=5
        --cem-receding-horizon=5
        --action-block=5
        --cem-num-samples=300
        --cem-iterations=30
        --cem-topk=30
        --cem-var-scale=1.0
        --cem-cost-mode=last
        --output="$result_dir/result.json"
      )
      python impls/eval_lewm_control_suite.py "${args[@]}" --validate-only
      python impls/eval_lewm_control_suite.py "${args[@]}" 2>&1 | tee "$result_dir/eval.log"
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
