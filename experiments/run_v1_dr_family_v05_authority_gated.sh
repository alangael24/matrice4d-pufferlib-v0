#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-drone}"
BATCH_ID="${BATCH_ID:-v1_norm_thrust_dr_family_v05_authority_gated_$(date -u +%Y%m%dT%H%M%SZ)}"
BATCH_ROOT="${BATCH_ROOT:-runs/$BATCH_ID}"
BASE="${BASE:-}"
SEEDS="${SEEDS:-42}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-300000000}"
GPU_ID="${GPU_ID:-0}"
BUILD_FIRST="${BUILD_FIRST:-0}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"

TOTAL_AGENTS="${TOTAL_AGENTS:-8192}"
NUM_BUFFERS="${NUM_BUFFERS:-1}"
NUM_THREADS="${NUM_THREADS:-32}"
NUM_DRONES="${NUM_DRONES:-8192}"

TARGET_DIST="${TARGET_DIST:-5.0}"
OOB_RADIUS="${OOB_RADIUS:-12.0}"
YAW_MULT="${YAW_MULT:-5.0}"
RESET_POS_SCALE="${RESET_POS_SCALE:-1.0}"
RESET_YAW_RANGE="${RESET_YAW_RANGE:-3.14159}"
RESET_VEL_MAX="${RESET_VEL_MAX:-0.2}"
ACTION_SCALE="${ACTION_SCALE:-1.0}"
ACTION_MODE="${ACTION_MODE:-1}"
NORMALIZED_THRUST_MIN="${NORMALIZED_THRUST_MIN:-0.0}"
NORMALIZED_THRUST_MAX="${NORMALIZED_THRUST_MAX:-0.85}"

DR_USABLE_T2W_MIN="${DR_USABLE_T2W_MIN:-2.2}"
DR_USABLE_T2W_MAX="${DR_USABLE_T2W_MAX:-3.8}"
DR_MASS_MIN="${DR_MASS_MIN:-0.85}"
DR_MASS_MAX="${DR_MASS_MAX:-1.15}"
DR_INERTIA_MIN="${DR_INERTIA_MIN:-0.70}"
DR_INERTIA_MAX="${DR_INERTIA_MAX:-1.40}"
DR_MOTOR_THRUST_MIN="${DR_MOTOR_THRUST_MIN:-0.90}"
DR_MOTOR_THRUST_MAX="${DR_MOTOR_THRUST_MAX:-1.10}"
DR_MOTOR_TAU_MIN="${DR_MOTOR_TAU_MIN:-0.08}"
DR_MOTOR_TAU_MAX="${DR_MOTOR_TAU_MAX:-0.20}"
DR_YAW_TORQUE_MIN="${DR_YAW_TORQUE_MIN:-0.80}"
DR_YAW_TORQUE_MAX="${DR_YAW_TORQUE_MAX:-1.25}"
DR_COM_XY="${DR_COM_XY:-0.015}"
DR_COM_Z="${DR_COM_Z:-0.010}"
DR_LINEAR_DRAG_MIN="${DR_LINEAR_DRAG_MIN:-0.50}"
DR_LINEAR_DRAG_MAX="${DR_LINEAR_DRAG_MAX:-1.50}"
DR_ANGULAR_DAMPING_MIN="${DR_ANGULAR_DAMPING_MIN:-0.50}"
DR_ANGULAR_DAMPING_MAX="${DR_ANGULAR_DAMPING_MAX:-1.50}"

if [[ -z "$BASE" || ! -f "$BASE" ]]; then
  cat >&2 <<'EOF'
Missing base checkpoint.

Set BASE to the selected B2 checkpoint, for example:
BASE=/workspace/matrice4d-pufferlib-v0/base_B2_normalized_narrow_seed42.bin \
  bash experiments/run_v1_dr_family_v05_authority_gated.sh
EOF
  exit 2
fi

mkdir -p "$BATCH_ROOT"
git rev-parse HEAD > "$BATCH_ROOT/git_commit.txt" || true
git status --short > "$BATCH_ROOT/git_status.txt" || true
git diff > "$BATCH_ROOT/git_diff.patch" || true

{
  echo "experiment=dr_family_v05_authority_gated"
  echo "base_checkpoint=$BASE"
  echo "batch_id=$BATCH_ID"
  echo "total_timesteps=$TOTAL_TIMESTEPS"
  echo "seeds=$SEEDS"
  echo "total_agents=$TOTAL_AGENTS"
  echo "num_buffers=$NUM_BUFFERS"
  echo "num_threads=$NUM_THREADS"
  echo "normalized_thrust_max=$NORMALIZED_THRUST_MAX"
  echo "usable_t2w_min=$DR_USABLE_T2W_MIN"
  echo "usable_t2w_max=$DR_USABLE_T2W_MAX"
  echo "mass_range=$DR_MASS_MIN,$DR_MASS_MAX"
  echo "inertia_range=$DR_INERTIA_MIN,$DR_INERTIA_MAX"
  echo "motor_thrust_scale_range=$DR_MOTOR_THRUST_MIN,$DR_MOTOR_THRUST_MAX"
  echo "motor_tau_range=$DR_MOTOR_TAU_MIN,$DR_MOTOR_TAU_MAX"
  echo "yaw_torque_scale_range=$DR_YAW_TORQUE_MIN,$DR_YAW_TORQUE_MAX"
  echo "com_xy=$DR_COM_XY"
  echo "com_z=$DR_COM_Z"
  echo "linear_drag_range=$DR_LINEAR_DRAG_MIN,$DR_LINEAR_DRAG_MAX"
  echo "angular_damping_range=$DR_ANGULAR_DAMPING_MIN,$DR_ANGULAR_DAMPING_MAX"
} > "$BATCH_ROOT/batch_metadata.env"

if [[ "$BUILD_FIRST" == "1" ]]; then
  bash build.sh "$ENV_NAME" 2>&1 | tee "$BATCH_ROOT/build_stdout.txt"
fi

write_command() {
  local out="$1"
  shift
  printf '%q ' "$@" > "$out"
  printf '\n' >> "$out"
  chmod +x "$out"
}

run_seed() {
  local seed="$1"
  local run_name="v1_norm_thrust_dr_family_v05_authority_gated_from_B2_seed${seed}_${TOTAL_TIMESTEPS}"
  local run_dir="$BATCH_ROOT/$run_name"
  mkdir -p "$run_dir/checkpoints" "$run_dir/logs"

  local cmd=(
    python -m pufferlib.pufferl train "$ENV_NAME"
    --tag "$run_name"
    --load-model-path "$BASE"
    --checkpoint-dir "$run_dir/checkpoints"
    --log-dir "$run_dir/logs"
    --checkpoint-interval "$CHECKPOINT_INTERVAL"
    --seed "$seed"
    --train.seed "$seed"
    --train.total-timesteps "$TOTAL_TIMESTEPS"
    --vec.total-agents "$TOTAL_AGENTS"
    --vec.num-buffers "$NUM_BUFFERS"
    --vec.num-threads "$NUM_THREADS"
    --env.num-drones "$NUM_DRONES"
    --env.hover-target-dist "$TARGET_DIST"
    --env.oob-radius "$OOB_RADIUS"
    --env.alpha-omega-z-mult "$YAW_MULT"
    --env.action-scale "$ACTION_SCALE"
    --env.action-mode "$ACTION_MODE"
    --env.normalized-thrust-min "$NORMALIZED_THRUST_MIN"
    --env.normalized-thrust-max "$NORMALIZED_THRUST_MAX"
    --env.reset-pos-scale "$RESET_POS_SCALE"
    --env.reset-yaw-range "$RESET_YAW_RANGE"
    --env.reset-vel-max "$RESET_VEL_MAX"
    --env.domain-randomization 1.0
    --env.dr-authority-gated 1.0
    --env.dr-usable-t2w-min "$DR_USABLE_T2W_MIN"
    --env.dr-usable-t2w-max "$DR_USABLE_T2W_MAX"
    --env.dr-mass-min "$DR_MASS_MIN"
    --env.dr-mass-max "$DR_MASS_MAX"
    --env.dr-inertia-min "$DR_INERTIA_MIN"
    --env.dr-inertia-max "$DR_INERTIA_MAX"
    --env.dr-motor-thrust-min "$DR_MOTOR_THRUST_MIN"
    --env.dr-motor-thrust-max "$DR_MOTOR_THRUST_MAX"
    --env.dr-motor-tau-min "$DR_MOTOR_TAU_MIN"
    --env.dr-motor-tau-max "$DR_MOTOR_TAU_MAX"
    --env.dr-yaw-torque-min "$DR_YAW_TORQUE_MIN"
    --env.dr-yaw-torque-max "$DR_YAW_TORQUE_MAX"
    --env.dr-com-xy "$DR_COM_XY"
    --env.dr-com-z "$DR_COM_Z"
    --env.dr-linear-drag-min "$DR_LINEAR_DRAG_MIN"
    --env.dr-linear-drag-max "$DR_LINEAR_DRAG_MAX"
    --env.dr-angular-damping-min "$DR_ANGULAR_DAMPING_MIN"
    --env.dr-angular-damping-max "$DR_ANGULAR_DAMPING_MAX"
    --policy.num-layers 3
  )

  write_command "$run_dir/command.sh" CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}"
  {
    echo "run_name=$run_name"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "seed=$seed"
    echo "base_checkpoint=$BASE"
  } > "$run_dir/run_metadata.env"

  echo "### Running $run_name"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" 2>&1 | tee "$run_dir/stdout.txt"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$run_dir/run_metadata.env"
  echo "exit_code=$exit_code" >> "$run_dir/run_metadata.env"

  if [[ -x scripts/find_checkpoints.sh ]]; then
    bash scripts/find_checkpoints.sh "$run_dir/checkpoints" > "$run_dir/checkpoint_summary.txt" || true
  fi
  if [[ -f scripts/summarize_runs.py ]]; then
    python scripts/summarize_runs.py --run-dir "$run_dir" --output "$run_dir/summary.json" || true
  fi

  if [[ "$exit_code" != "0" ]]; then
    echo "Run failed: $run_name (exit $exit_code)" | tee -a "$BATCH_ROOT/failures.txt"
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
      exit "$exit_code"
    fi
  fi
}

for seed in $SEEDS; do
  run_seed "$seed"
done
