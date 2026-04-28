#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BATCH_ID="${BATCH_ID:-v0_11_dr_ablation_target5_$(date -u +%Y%m%dT%H%M%SZ)}"
BATCH_ROOT="${BATCH_ROOT:-runs/${BATCH_ID}}"
ARCHIVE_DIR="${ARCHIVE_DIR:-artifacts}"
GPU_ID="${GPU_ID:-0}"
ENV_NAME="${ENV_NAME:-drone}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
BUILD_FIRST="${BUILD_FIRST:-1}"

SEEDS="${SEEDS:-42 43 44 45 46}"
BASE_SEED="${BASE_SEED:-46}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"

TOTAL_AGENTS="${TOTAL_AGENTS:-4096}"
NUM_BUFFERS="${NUM_BUFFERS:-8}"
NUM_THREADS="${NUM_THREADS:-32}"
NUM_DRONES="${NUM_DRONES:-64}"

TARGET_DIST="${TARGET_DIST:-5}"
ACTION_SCALE="${ACTION_SCALE:-0.5}"
YAW_MULT="${YAW_MULT:-5}"
DOMAIN_RANDOMIZATION="${DOMAIN_RANDOMIZATION:-1}"
RESET_YAW_RANGE="${RESET_YAW_RANGE:-3.14159}"
RESET_VEL_MAX="${RESET_VEL_MAX:-0.2}"
RESET_POS_SCALE="${RESET_POS_SCALE:-1.0}"
OOB_RADIUS="${OOB_RADIUS:-12}"

# Default sweep is intentionally small: these are the most likely bottlenecks.
# Use ABLATIONS=all to run every group.
ABLATIONS="${ABLATIONS:-k_thrust motor_lag latency_noise}"
if [[ "$ABLATIONS" == "all" ]]; then
  ABLATIONS="light_control k_thrust motor_lag latency_noise com inertia drag mass medium_all"
fi

# DR-light baseline values: known to pass target5 in V0.11.
LIGHT_DR_MASS="${LIGHT_DR_MASS:-0.05}"
LIGHT_DR_INERTIA="${LIGHT_DR_INERTIA:-0.10}"
LIGHT_DR_K_THRUST="${LIGHT_DR_K_THRUST:-0.10}"
LIGHT_DR_LINEAR_DRAG="${LIGHT_DR_LINEAR_DRAG:-0.20}"
LIGHT_DR_YAW_DRAG="${LIGHT_DR_YAW_DRAG:-0.20}"
LIGHT_DR_MOTOR_LAG="${LIGHT_DR_MOTOR_LAG:-0.05}"
LIGHT_DR_COM_XY="${LIGHT_DR_COM_XY:-0.01}"
LIGHT_DR_COM_Z="${LIGHT_DR_COM_Z:-0.0}"
LIGHT_ACTION_LATENCY="${LIGHT_ACTION_LATENCY:-0.0}"
LIGHT_SENSOR_NOISE="${LIGHT_SENSOR_NOISE:-0.0}"

# DR-medium values: known to fail target5 when applied all together.
MEDIUM_DR_MASS="${MEDIUM_DR_MASS:-0.10}"
MEDIUM_DR_INERTIA="${MEDIUM_DR_INERTIA:-0.20}"
MEDIUM_DR_K_THRUST="${MEDIUM_DR_K_THRUST:-0.20}"
MEDIUM_DR_LINEAR_DRAG="${MEDIUM_DR_LINEAR_DRAG:-0.40}"
MEDIUM_DR_YAW_DRAG="${MEDIUM_DR_YAW_DRAG:-0.40}"
MEDIUM_DR_MOTOR_LAG="${MEDIUM_DR_MOTOR_LAG:-0.15}"
MEDIUM_DR_COM_XY="${MEDIUM_DR_COM_XY:-0.02}"
MEDIUM_DR_COM_Z="${MEDIUM_DR_COM_Z:-0.03}"
MEDIUM_ACTION_LATENCY="${MEDIUM_ACTION_LATENCY:-0.01}"
MEDIUM_SENSOR_NOISE="${MEDIUM_SENSOR_NOISE:-0.01}"

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
      | grep 'v0_10' \
      | grep 'target5' \
      | grep "seed${BASE_SEED}" \
      | grep 'scale0.5' \
      | grep 'yawMult5' \
      | sort \
      | tail -1 || true
  )"
fi

if [[ -z "$BASE" || ! -f "$BASE" ]]; then
  cat >&2 <<EOF
Missing base checkpoint.

Set BASE explicitly, for example:
BASE=/matrice4d-pufferlib-v0/base_target5_seed46.bin \\
  bash experiments/run_v0_11_dr_ablation.sh

This sweep is intended to continue from the V0.10 target5 nominal checkpoint.
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
  echo "ablations=$ABLATIONS"
  echo "total_timesteps=$TOTAL_TIMESTEPS"
  echo "target_dist=$TARGET_DIST"
  echo "action_scale=$ACTION_SCALE"
  echo "yaw_mult=$YAW_MULT"
  echo "domain_randomization=$DOMAIN_RANDOMIZATION"
  echo "light_dr_mass=$LIGHT_DR_MASS"
  echo "light_dr_inertia=$LIGHT_DR_INERTIA"
  echo "light_dr_k_thrust=$LIGHT_DR_K_THRUST"
  echo "light_dr_linear_drag=$LIGHT_DR_LINEAR_DRAG"
  echo "light_dr_yaw_drag=$LIGHT_DR_YAW_DRAG"
  echo "light_dr_motor_lag=$LIGHT_DR_MOTOR_LAG"
  echo "light_dr_com_xy=$LIGHT_DR_COM_XY"
  echo "light_dr_com_z=$LIGHT_DR_COM_Z"
  echo "light_action_latency=$LIGHT_ACTION_LATENCY"
  echo "light_sensor_noise=$LIGHT_SENSOR_NOISE"
  echo "medium_dr_mass=$MEDIUM_DR_MASS"
  echo "medium_dr_inertia=$MEDIUM_DR_INERTIA"
  echo "medium_dr_k_thrust=$MEDIUM_DR_K_THRUST"
  echo "medium_dr_linear_drag=$MEDIUM_DR_LINEAR_DRAG"
  echo "medium_dr_yaw_drag=$MEDIUM_DR_YAW_DRAG"
  echo "medium_dr_motor_lag=$MEDIUM_DR_MOTOR_LAG"
  echo "medium_dr_com_xy=$MEDIUM_DR_COM_XY"
  echo "medium_dr_com_z=$MEDIUM_DR_COM_Z"
  echo "medium_action_latency=$MEDIUM_ACTION_LATENCY"
  echo "medium_sensor_noise=$MEDIUM_SENSOR_NOISE"
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

configure_ablation() {
  local ablation="$1"

  DR_MASS="$LIGHT_DR_MASS"
  DR_INERTIA="$LIGHT_DR_INERTIA"
  DR_K_THRUST="$LIGHT_DR_K_THRUST"
  DR_LINEAR_DRAG="$LIGHT_DR_LINEAR_DRAG"
  DR_YAW_DRAG="$LIGHT_DR_YAW_DRAG"
  DR_MOTOR_LAG="$LIGHT_DR_MOTOR_LAG"
  DR_COM_XY="$LIGHT_DR_COM_XY"
  DR_COM_Z="$LIGHT_DR_COM_Z"
  ACTION_LATENCY="$LIGHT_ACTION_LATENCY"
  SENSOR_NOISE="$LIGHT_SENSOR_NOISE"

  case "$ablation" in
    light_control)
      ;;
    mass)
      DR_MASS="$MEDIUM_DR_MASS"
      ;;
    inertia)
      DR_INERTIA="$MEDIUM_DR_INERTIA"
      ;;
    k_thrust|thrust|kthrust)
      DR_K_THRUST="$MEDIUM_DR_K_THRUST"
      ;;
    drag)
      DR_LINEAR_DRAG="$MEDIUM_DR_LINEAR_DRAG"
      DR_YAW_DRAG="$MEDIUM_DR_YAW_DRAG"
      ;;
    motor_lag|lag)
      DR_MOTOR_LAG="$MEDIUM_DR_MOTOR_LAG"
      ;;
    com)
      DR_COM_XY="$MEDIUM_DR_COM_XY"
      DR_COM_Z="$MEDIUM_DR_COM_Z"
      ;;
    latency_noise)
      ACTION_LATENCY="$MEDIUM_ACTION_LATENCY"
      SENSOR_NOISE="$MEDIUM_SENSOR_NOISE"
      ;;
    latency)
      ACTION_LATENCY="$MEDIUM_ACTION_LATENCY"
      ;;
    sensor_noise|noise)
      SENSOR_NOISE="$MEDIUM_SENSOR_NOISE"
      ;;
    medium_all)
      DR_MASS="$MEDIUM_DR_MASS"
      DR_INERTIA="$MEDIUM_DR_INERTIA"
      DR_K_THRUST="$MEDIUM_DR_K_THRUST"
      DR_LINEAR_DRAG="$MEDIUM_DR_LINEAR_DRAG"
      DR_YAW_DRAG="$MEDIUM_DR_YAW_DRAG"
      DR_MOTOR_LAG="$MEDIUM_DR_MOTOR_LAG"
      DR_COM_XY="$MEDIUM_DR_COM_XY"
      DR_COM_Z="$MEDIUM_DR_COM_Z"
      ACTION_LATENCY="$MEDIUM_ACTION_LATENCY"
      SENSOR_NOISE="$MEDIUM_SENSOR_NOISE"
      ;;
    *)
      echo "Unknown ablation: $ablation" >&2
      echo "Valid: light_control mass inertia k_thrust drag motor_lag com latency_noise latency sensor_noise medium_all all" >&2
      exit 2
      ;;
  esac
}

run_experiment() {
  local ablation="$1"
  local seed="$2"

  configure_ablation "$ablation"

  local safe_ablation="${ablation//[^A-Za-z0-9_.-]/_}"
  local run_name="v0_11_target5_dr_ablation_${safe_ablation}_seed${seed}_${TOTAL_TIMESTEPS}"
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
    echo "ablation=$ablation"
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

for ablation in $ABLATIONS; do
  for seed in $SEEDS; do
    run_experiment "$ablation" "$seed"
  done
done

archive_batch
