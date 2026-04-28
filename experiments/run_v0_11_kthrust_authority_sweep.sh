#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BATCH_ID="${BATCH_ID:-v0_11_kthrust_authority_$(date -u +%Y%m%dT%H%M%SZ)}"
BATCH_ROOT="${BATCH_ROOT:-runs/${BATCH_ID}}"
ARCHIVE_DIR="${ARCHIVE_DIR:-artifacts}"
GPU_ID="${GPU_ID:-0}"
ENV_NAME="${ENV_NAME:-drone}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
BUILD_FIRST="${BUILD_FIRST:-1}"

SEEDS="${SEEDS:-42 43 44 45 46}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
CASES="${CASES:-kt15_scale05 kt20_scale05 kt20_scale07 kt20_scale08}"

TOTAL_AGENTS="${TOTAL_AGENTS:-4096}"
NUM_BUFFERS="${NUM_BUFFERS:-8}"
NUM_THREADS="${NUM_THREADS:-32}"
NUM_DRONES="${NUM_DRONES:-64}"

TARGET_DIST="${TARGET_DIST:-5}"
YAW_MULT="${YAW_MULT:-5}"
DOMAIN_RANDOMIZATION="${DOMAIN_RANDOMIZATION:-1}"
RESET_YAW_RANGE="${RESET_YAW_RANGE:-3.14159}"
RESET_VEL_MAX="${RESET_VEL_MAX:-0.2}"
RESET_POS_SCALE="${RESET_POS_SCALE:-1.0}"
OOB_RADIUS="${OOB_RADIUS:-12}"

# Keep all non-thrust DR at the DR-light baseline.
DR_MASS="${DR_MASS:-0.05}"
DR_INERTIA="${DR_INERTIA:-0.10}"
DR_LINEAR_DRAG="${DR_LINEAR_DRAG:-0.20}"
DR_YAW_DRAG="${DR_YAW_DRAG:-0.20}"
DR_MOTOR_LAG="${DR_MOTOR_LAG:-0.05}"
DR_COM_XY="${DR_COM_XY:-0.01}"
DR_COM_Z="${DR_COM_Z:-0.0}"
ACTION_LATENCY="${ACTION_LATENCY:-0.0}"
SENSOR_NOISE="${SENSOR_NOISE:-0.0}"

BASE="${BASE:-}"
read -r -a EXTRA_ARGV <<< "${EXTRA_ARGS:-}"

mkdir -p "$BATCH_ROOT" "$ARCHIVE_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Could not find python or python3" >&2
    exit 1
  fi
fi

if [[ -z "$BASE" ]]; then
  BASE="$(
    find runs -type f -name '*.bin' 2>/dev/null \
      | grep 'dr_light' \
      | grep 'seed46' \
      | sort \
      | tail -1 || true
  )"
fi

if [[ -z "$BASE" || ! -f "$BASE" ]]; then
  cat >&2 <<EOF
Missing base checkpoint.

Set BASE explicitly, for example:
BASE=/matrice4d-pufferlib-v0/base_dr_light_seed46.bin \\
  bash experiments/run_v0_11_kthrust_authority_sweep.sh

This sweep is intended to continue from the V0.11 DR-light target5 baseline.
EOF
  exit 1
fi

ARCHIVED=0
archive_batch() {
  if [[ "$ARCHIVED" == "0" && -d "$BATCH_ROOT" ]]; then
    "$PYTHON_BIN" scripts/summarize_runs.py --runs-root "$BATCH_ROOT" \
      --output "$BATCH_ROOT/summary.json" \
      --csv "$BATCH_ROOT/summary.csv" || true
    bash scripts/archive_artifacts.sh "$BATCH_ROOT" "$ARCHIVE_DIR"
    ARCHIVED=1
  fi
}

trap 'status=$?; if [[ "$status" != "0" ]]; then archive_batch || true; fi' EXIT

git rev-parse HEAD > "$BATCH_ROOT/git_commit.txt"
git status --short --branch > "$BATCH_ROOT/git_status.txt"
git diff > "$BATCH_ROOT/git_diff.patch"

{
  echo "batch_id=$BATCH_ID"
  echo "base_checkpoint=$BASE"
  echo "seeds=$SEEDS"
  echo "cases=$CASES"
  echo "total_timesteps=$TOTAL_TIMESTEPS"
  echo "target_dist=$TARGET_DIST"
  echo "yaw_mult=$YAW_MULT"
  echo "domain_randomization=$DOMAIN_RANDOMIZATION"
  echo "dr_mass=$DR_MASS"
  echo "dr_inertia=$DR_INERTIA"
  echo "dr_linear_drag=$DR_LINEAR_DRAG"
  echo "dr_yaw_drag=$DR_YAW_DRAG"
  echo "dr_motor_lag=$DR_MOTOR_LAG"
  echo "dr_com_xy=$DR_COM_XY"
  echo "dr_com_z=$DR_COM_Z"
  echo "action_latency=$ACTION_LATENCY"
  echo "sensor_noise=$SENSOR_NOISE"
} > "$BATCH_ROOT/batch_metadata.env"

if [[ "$BUILD_FIRST" == "1" ]]; then
  {
    echo "### build.sh drone"
    bash build.sh drone
  } 2>&1 | tee "$BATCH_ROOT/build_stdout.txt"
fi

write_command() {
  local out="$1"
  shift
  printf '%q ' "$@" > "$out"
  printf '\n' >> "$out"
  chmod +x "$out"
}

configure_case() {
  local case_name="$1"
  case "$case_name" in
    kt15_scale05)
      DR_K_THRUST="0.15"
      ACTION_SCALE="0.5"
      ;;
    kt20_scale05)
      DR_K_THRUST="0.20"
      ACTION_SCALE="0.5"
      ;;
    kt20_scale07)
      DR_K_THRUST="0.20"
      ACTION_SCALE="0.7"
      ;;
    kt20_scale08)
      DR_K_THRUST="0.20"
      ACTION_SCALE="0.8"
      ;;
    *)
      echo "Unknown case: $case_name" >&2
      echo "Valid cases: kt15_scale05 kt20_scale05 kt20_scale07 kt20_scale08" >&2
      exit 2
      ;;
  esac
}

run_experiment() {
  local case_name="$1"
  local seed="$2"
  configure_case "$case_name"

  local run_name="v0_11_kthrust_${case_name}_seed${seed}_${TOTAL_TIMESTEPS}"
  local run_dir="${BATCH_ROOT}/${run_name}"
  mkdir -p "$run_dir/checkpoints" "$run_dir/logs"

  cp "$BATCH_ROOT/git_commit.txt" "$run_dir/git_commit.txt"
  cp "$BATCH_ROOT/git_status.txt" "$run_dir/git_status.txt"
  cp "$BATCH_ROOT/git_diff.patch" "$run_dir/git_diff.patch"

  local cmd=(
    puffer train "$ENV_NAME"
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
    --env.domain-randomization "$DOMAIN_RANDOMIZATION"
    --env.action-scale "$ACTION_SCALE"
    --env.reset-yaw-range "$RESET_YAW_RANGE"
    --env.reset-vel-max "$RESET_VEL_MAX"
    --env.reset-pos-scale "$RESET_POS_SCALE"
    --env.oob-radius "$OOB_RADIUS"
    --env.alpha-omega-z-mult "$YAW_MULT"
    --env.dr-mass "$DR_MASS"
    --env.dr-inertia "$DR_INERTIA"
    --env.dr-k-thrust "$DR_K_THRUST"
    --env.dr-linear-drag "$DR_LINEAR_DRAG"
    --env.dr-yaw-drag "$DR_YAW_DRAG"
    --env.dr-motor-lag "$DR_MOTOR_LAG"
    --env.dr-com-xy "$DR_COM_XY"
    --env.dr-com-z "$DR_COM_Z"
    --env.action-latency "$ACTION_LATENCY"
    --env.sensor-noise "$SENSOR_NOISE"
    --policy.num-layers 3
  )

  if [[ ${#EXTRA_ARGV[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGV[@]}")
  fi

  write_command "$run_dir/command.sh" CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}"

  {
    echo "run_name=$run_name"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "gpu_id=$GPU_ID"
    echo "base_checkpoint=$BASE"
    echo "sweep_case=$case_name"
    echo "seed=$seed"
    echo "total_timesteps=$TOTAL_TIMESTEPS"
    echo "target_dist=$TARGET_DIST"
    echo "action_scale=$ACTION_SCALE"
    echo "yaw_mult=$YAW_MULT"
    echo "domain_randomization=$DOMAIN_RANDOMIZATION"
    echo "reset_yaw_range=$RESET_YAW_RANGE"
    echo "reset_vel_max=$RESET_VEL_MAX"
    echo "reset_pos_scale=$RESET_POS_SCALE"
    echo "oob_radius=$OOB_RADIUS"
    echo "dr_mass=$DR_MASS"
    echo "dr_inertia=$DR_INERTIA"
    echo "dr_k_thrust=$DR_K_THRUST"
    echo "dr_linear_drag=$DR_LINEAR_DRAG"
    echo "dr_yaw_drag=$DR_YAW_DRAG"
    echo "dr_motor_lag=$DR_MOTOR_LAG"
    echo "dr_com_xy=$DR_COM_XY"
    echo "dr_com_z=$DR_COM_Z"
    echo "action_latency=$ACTION_LATENCY"
    echo "sensor_noise=$SENSOR_NOISE"
  } > "$run_dir/run_metadata.env"

  echo
  echo "### Running $run_name"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" 2>&1 | tee "$run_dir/stdout.txt"
  local exit_code="${PIPESTATUS[0]}"
  set -e

  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$run_dir/run_metadata.env"
  echo "exit_code=$exit_code" >> "$run_dir/run_metadata.env"

  bash scripts/find_checkpoints.sh "$run_dir/checkpoints" > "$run_dir/checkpoint_summary.txt" || true
  "$PYTHON_BIN" scripts/summarize_runs.py --run-dir "$run_dir" --output "$run_dir/summary.json" || true

  if [[ "$exit_code" != "0" ]]; then
    echo "Run failed: $run_name (exit $exit_code)" | tee -a "$BATCH_ROOT/failures.txt"
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
      archive_batch
      exit "$exit_code"
    fi
  fi
}

for case_name in $CASES; do
  for seed in $SEEDS; do
    run_experiment "$case_name" "$seed"
  done
done

archive_batch
