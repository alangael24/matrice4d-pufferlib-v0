#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BATCH_ID="${BATCH_ID:-v0_1_$(date -u +%Y%m%dT%H%M%SZ)}"
BATCH_ROOT="${BATCH_ROOT:-runs/${BATCH_ID}}"
ARCHIVE_DIR="${ARCHIVE_DIR:-artifacts}"
GPU_ID="${GPU_ID:-0}"
ENV_NAME="${ENV_NAME:-drone}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
BUILD_FIRST="${BUILD_FIRST:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINT_TEST_TIMESTEPS="${CHECKPOINT_TEST_TIMESTEPS:-100000}"
TOTAL_TIMESTEPS_PHASEA="${TOTAL_TIMESTEPS_PHASEA:-30000000}"
TOTAL_TIMESTEPS_TARGET2="${TOTAL_TIMESTEPS_TARGET2:-30000000}"
TOTAL_TIMESTEPS_TARGET5="${TOTAL_TIMESTEPS_TARGET5:-30000000}"

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

ARCHIVED=0
archive_batch() {
  if [[ "$ARCHIVED" == "0" && -d "$BATCH_ROOT" ]]; then
    bash scripts/archive_artifacts.sh "$BATCH_ROOT" "$ARCHIVE_DIR"
    ARCHIVED=1
  fi
}

trap 'status=$?; if [[ "$status" != "0" ]]; then archive_batch || true; fi' EXIT

git rev-parse HEAD > "$BATCH_ROOT/git_commit.txt"
git status --short --branch > "$BATCH_ROOT/git_status.txt"
git diff > "$BATCH_ROOT/git_diff.patch"

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

run_experiment() {
  local name="$1"
  local timesteps="$2"
  local seed="$3"
  local target_dist="$4"
  local action_scale="$5"
  local domain_randomization="$6"
  local reset_yaw_range="$7"
  local reset_vel_max="$8"
  local reset_pos_scale="$9"

  local run_dir="${BATCH_ROOT}/${name}"
  mkdir -p "$run_dir/checkpoints" "$run_dir/logs"

  cp "$BATCH_ROOT/git_commit.txt" "$run_dir/git_commit.txt"
  cp "$BATCH_ROOT/git_status.txt" "$run_dir/git_status.txt"
  cp "$BATCH_ROOT/git_diff.patch" "$run_dir/git_diff.patch"

  local cmd=(
    puffer train "$ENV_NAME"
    --tag "$name"
    --checkpoint-dir "$run_dir/checkpoints"
    --log-dir "$run_dir/logs"
    --checkpoint-interval "$CHECKPOINT_INTERVAL"
    --seed "$seed"
    --train.seed "$seed"
    --train.total-timesteps "$timesteps"
    --env.hover-target-dist "$target_dist"
    --env.domain-randomization "$domain_randomization"
    --env.action-scale "$action_scale"
    --env.reset-yaw-range "$reset_yaw_range"
    --env.reset-vel-max "$reset_vel_max"
    --env.reset-pos-scale "$reset_pos_scale"
    --policy.num-layers 3
  )

  if [[ ${#EXTRA_ARGV[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGV[@]}")
  fi

  write_command "$run_dir/command.sh" CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}"

  {
    echo "run_name=$name"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "gpu_id=$GPU_ID"
    echo "timesteps=$timesteps"
    echo "seed=$seed"
    echo "target_dist=$target_dist"
    echo "action_scale=$action_scale"
    echo "domain_randomization=$domain_randomization"
    echo "reset_yaw_range=$reset_yaw_range"
    echo "reset_vel_max=$reset_vel_max"
    echo "reset_pos_scale=$reset_pos_scale"
  } > "$run_dir/run_metadata.env"

  echo
  echo "### Running $name"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" 2>&1 | tee "$run_dir/stdout.txt"
  local exit_code="${PIPESTATUS[0]}"
  set -e

  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$run_dir/run_metadata.env"
  echo "exit_code=$exit_code" >> "$run_dir/run_metadata.env"

  bash scripts/find_checkpoints.sh "$run_dir/checkpoints" > "$run_dir/checkpoint_summary.txt" || true
  "$PYTHON_BIN" scripts/summarize_runs.py --run-dir "$run_dir" --output "$run_dir/summary.json" || true

  if [[ "$exit_code" != "0" ]]; then
    echo "Run failed: $name (exit $exit_code)" | tee -a "$BATCH_ROOT/failures.txt"
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
      bash scripts/archive_artifacts.sh "$BATCH_ROOT" "$ARCHIVE_DIR"
      exit "$exit_code"
    fi
  fi
}

# 1. Checkpoint smoke test, using the same nominal reset as Phase A.
run_experiment \
  "checkpoint_test_seed46_target05_scale02" \
  "$CHECKPOINT_TEST_TIMESTEPS" \
  46 \
  0.5 \
  0.2 \
  0 \
  0 \
  0 \
  0

# 2. Phase A: real close-hover baseline, no reset randomization.
run_experiment \
  "phaseA_close_seed46_30m" \
  "$TOTAL_TIMESTEPS_PHASEA" \
  46 \
  0.5 \
  0.2 \
  0 \
  0 \
  0 \
  0

# 3. Phase B: target 2 m seeds 42-46 with hard reset variation.
for seed in 42 43 44 45 46; do
  run_experiment \
    "target2m_seed${seed}_30m" \
    "$TOTAL_TIMESTEPS_TARGET2" \
    "$seed" \
    2 \
    0.2 \
    0 \
    3.14159 \
    0.2 \
    1.0
done

# 4. Phase B: target 5 m action-scale sweep with hard reset variation.
for action_scale in 0.2 0.3 0.4 0.5; do
  safe_scale="${action_scale/./}"
  run_experiment \
    "target5m_seed42_scale${safe_scale}_30m" \
    "$TOTAL_TIMESTEPS_TARGET5" \
    42 \
    5 \
    "$action_scale" \
    0 \
    3.14159 \
    0.2 \
    1.0
done

"$PYTHON_BIN" scripts/summarize_runs.py --runs-root "$BATCH_ROOT" \
  --output "$BATCH_ROOT/summary.json" \
  --csv "$BATCH_ROOT/summary.csv"

archive_batch
