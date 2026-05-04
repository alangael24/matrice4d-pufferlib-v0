#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BENCH_ID="${BENCH_ID:-cuda_env_benchmark_$(date -u +%Y%m%dT%H%M%SZ)}"
BENCH_ROOT="${BENCH_ROOT:-runs/${BENCH_ID}}"
ARCHIVE_DIR="${ARCHIVE_DIR:-artifacts}"
ARCHIVE="${ARCHIVE:-1}"
ENV_NAME="${ENV_NAME:-drone}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PUFFER_BIN="${PUFFER_BIN:-puffer}"
GPU_ID="${GPU_ID:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"

RUN_CORRECTNESS="${RUN_CORRECTNESS:-1}"
RUN_SPEED="${RUN_SPEED:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_CPU_TRAIN="${RUN_CPU_TRAIN:-1}"
RUN_CUDA_TRAIN="${RUN_CUDA_TRAIN:-1}"

CUDA_CHECK_STEPS="${CUDA_CHECK_STEPS:-200}"
CUDA_CHECK_AGENTS="${CUDA_CHECK_AGENTS:-128}"
CUDA_CHECK_ACTION_AMPLITUDE="${CUDA_CHECK_ACTION_AMPLITUDE:-0.05}"

PROFILE_BUFFERS="${PROFILE_BUFFERS:-8}"
PROFILE_THREADS="${PROFILE_THREADS:-32}"
PROFILE_HORIZON="${PROFILE_HORIZON:-256}"
PROFILE_TOTAL_AGENTS="${PROFILE_TOTAL_AGENTS:-32768}"

SEEDS="${SEEDS:-42}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3000000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
TOTAL_AGENTS="${TOTAL_AGENTS:-4096}"
NUM_BUFFERS="${NUM_BUFFERS:-8}"
NUM_THREADS="${NUM_THREADS:-32}"
NUM_DRONES="${NUM_DRONES:-64}"

read -r -a EXTRA_TRAIN_ARGV <<< "${EXTRA_TRAIN_ARGS:-}"

export TAR_OPTIONS="${TAR_OPTIONS:---no-same-owner}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Could not find python or python3" >&2
    exit 1
  fi
fi

mkdir -p "$BENCH_ROOT/build" "$BENCH_ROOT/correctness" "$BENCH_ROOT/speed" "$BENCH_ROOT/train" "$ARCHIVE_DIR"

write_command() {
  local out="$1"
  shift
  printf '%q ' "$@" > "$out"
  printf '\n' >> "$out"
  chmod +x "$out"
}

run_logged() {
  local label="$1"
  local out="$2"
  shift 2

  mkdir -p "$(dirname "$out")"
  write_command "${out}.cmd" "$@"

  echo
  echo "### $label"
  set +e
  "$@" 2>&1 | tee "$out"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "$status" > "${out}.exit_code"

  if [[ "$status" != "0" ]]; then
    echo "Failed: $label (exit $status)" | tee -a "$BENCH_ROOT/failures.txt"
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
      finalize
      exit "$status"
    fi
  fi
}

write_root_metadata() {
  git rev-parse HEAD > "$BENCH_ROOT/git_commit.txt"
  git status --short --branch > "$BENCH_ROOT/git_status.txt"
  git diff > "$BENCH_ROOT/git_diff.patch"

  {
    echo "bench_id=$BENCH_ID"
    echo "env_name=$ENV_NAME"
    echo "gpu_id=$GPU_ID"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "run_correctness=$RUN_CORRECTNESS"
    echo "run_speed=$RUN_SPEED"
    echo "run_train=$RUN_TRAIN"
    echo "run_cpu_train=$RUN_CPU_TRAIN"
    echo "run_cuda_train=$RUN_CUDA_TRAIN"
    echo "cuda_check_steps=$CUDA_CHECK_STEPS"
    echo "cuda_check_agents=$CUDA_CHECK_AGENTS"
    echo "profile_buffers=$PROFILE_BUFFERS"
    echo "profile_threads=$PROFILE_THREADS"
    echo "profile_horizon=$PROFILE_HORIZON"
    echo "profile_total_agents=$PROFILE_TOTAL_AGENTS"
    echo "seeds=$SEEDS"
    echo "total_timesteps=$TOTAL_TIMESTEPS"
    echo "total_agents=$TOTAL_AGENTS"
    echo "num_buffers=$NUM_BUFFERS"
    echo "num_threads=$NUM_THREADS"
    echo "num_drones=$NUM_DRONES"
    echo "success_correctness=matrice4d_v0_checks passed and cuda_checks passed"
    echo "success_speed=cuda envspeed throughput greater than cpu envspeed throughput"
    echo "success_quality=cuda reaches similar env metrics in lower wall-clock time"
  } > "$BENCH_ROOT/benchmark_metadata.env"
}

summarize_train_runs() {
  if [[ -d "$BENCH_ROOT/train" ]]; then
    "$PYTHON_BIN" scripts/summarize_runs.py \
      --runs-root "$BENCH_ROOT/train" \
      --output "$BENCH_ROOT/train_summary.json" \
      --csv "$BENCH_ROOT/train_summary.csv" || true
  fi
}

ARCHIVED=0
finalize() {
  if [[ "$ARCHIVED" == "1" ]]; then
    return
  fi

  summarize_train_runs
  "$PYTHON_BIN" tools/summarize_cuda_benchmark.py \
    --bench-root "$BENCH_ROOT" \
    --output "$BENCH_ROOT/summary.json" \
    --markdown "$BENCH_ROOT/summary.md" || true

  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BENCH_ROOT/benchmark_metadata.env"

  if [[ "$ARCHIVE" == "1" ]]; then
    bash scripts/archive_artifacts.sh "$BENCH_ROOT" "$ARCHIVE_DIR" || true
  fi
  ARCHIVED=1
}

trap 'status=$?; if [[ "$status" != "0" ]]; then finalize || true; fi' EXIT

run_correctness() {
  run_logged "Matrice 4D V0 CPU checks" \
    "$BENCH_ROOT/correctness/matrice4d_v0_checks.txt" \
    "$PYTHON_BIN" ocean/drone/matrice4d_v0_checks.py

  run_logged "Build CUDA env" \
    "$BENCH_ROOT/build/build_cuda.txt" \
    bash build.sh "$ENV_NAME"

  run_logged "CPU vs CUDA numeric parity" \
    "$BENCH_ROOT/correctness/cuda_checks.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" ocean/drone/cuda_checks.py \
      --steps "$CUDA_CHECK_STEPS" \
      --num-agents "$CUDA_CHECK_AGENTS" \
      --action-amplitude "$CUDA_CHECK_ACTION_AMPLITUDE"
}

run_speed() {
  run_logged "Build CPU-env profile baseline" \
    "$BENCH_ROOT/build/build_cpu_profile.txt" \
    env PUFFERLIB_DISABLE_ENV_CUDA=1 bash build.sh "$ENV_NAME" --profile

  run_logged "Envspeed CPU env plus GPU copies" \
    "$BENCH_ROOT/speed/envspeed_cpu_env.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" ./profile envspeed \
      --buffers "$PROFILE_BUFFERS" \
      --threads "$PROFILE_THREADS" \
      --horizon "$PROFILE_HORIZON" \
      --total-agents "$PROFILE_TOTAL_AGENTS"

  run_logged "Build CUDA-env profile" \
    "$BENCH_ROOT/build/build_cuda_profile.txt" \
    bash build.sh "$ENV_NAME" --profile

  run_logged "Envspeed CUDA env" \
    "$BENCH_ROOT/speed/envspeed_cuda_env.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" ./profile envspeed \
      --buffers "$PROFILE_BUFFERS" \
      --threads "$PROFILE_THREADS" \
      --horizon "$PROFILE_HORIZON" \
      --total-agents "$PROFILE_TOTAL_AGENTS"
}

run_train_one() {
  local backend="$1"
  local seed="$2"
  local run_name="${BENCH_ID}_${backend}_seed${seed}_${TOTAL_TIMESTEPS}"
  local run_dir="$BENCH_ROOT/train/$run_name"
  mkdir -p "$run_dir/checkpoints" "$run_dir/logs"

  cp "$BENCH_ROOT/git_commit.txt" "$run_dir/git_commit.txt"
  cp "$BENCH_ROOT/git_status.txt" "$run_dir/git_status.txt"
  cp "$BENCH_ROOT/git_diff.patch" "$run_dir/git_diff.patch"

  {
    echo "run_name=$run_name"
    echo "backend=$backend"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "gpu_id=$GPU_ID"
    echo "seed=$seed"
    echo "total_timesteps=$TOTAL_TIMESTEPS"
    echo "total_agents=$TOTAL_AGENTS"
    echo "num_buffers=$NUM_BUFFERS"
    echo "num_threads=$NUM_THREADS"
    echo "num_drones=$NUM_DRONES"
  } > "$run_dir/run_metadata.env"

  local cmd=(
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PUFFER_BIN" train "$ENV_NAME"
    --tag "$run_name"
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
    --profile True
  )

  if [[ "$backend" == "cpu_env" ]]; then
    cmd=(env CUDA_VISIBLE_DEVICES="$GPU_ID" PUFFERLIB_DISABLE_ENV_CUDA=1 "$PUFFER_BIN" train "$ENV_NAME"
      --tag "$run_name"
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
      --profile True
    )
  fi

  if [[ ${#EXTRA_TRAIN_ARGV[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_TRAIN_ARGV[@]}")
  fi

  write_command "$run_dir/command.sh" "${cmd[@]}"
  run_logged "Training $backend seed $seed" "$run_dir/stdout.txt" "${cmd[@]}"

  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$run_dir/run_metadata.env"
  cat "$run_dir/stdout.txt.exit_code" | awk '{print "exit_code="$1}' >> "$run_dir/run_metadata.env"

  bash scripts/find_checkpoints.sh "$run_dir/checkpoints" > "$run_dir/checkpoint_summary.txt" || true
  "$PYTHON_BIN" scripts/summarize_runs.py --run-dir "$run_dir" --output "$run_dir/summary.json" || true
}

run_training() {
  if [[ "$RUN_CPU_TRAIN" == "1" ]]; then
    run_logged "Build CPU env for training" \
      "$BENCH_ROOT/build/build_cpu_train.txt" \
      env PUFFERLIB_DISABLE_ENV_CUDA=1 bash build.sh "$ENV_NAME"

    for seed in $SEEDS; do
      run_train_one "cpu_env" "$seed"
    done
  fi

  if [[ "$RUN_CUDA_TRAIN" == "1" ]]; then
    run_logged "Build CUDA env for training" \
      "$BENCH_ROOT/build/build_cuda_train.txt" \
      bash build.sh "$ENV_NAME"

    for seed in $SEEDS; do
      run_train_one "cuda_env" "$seed"
    done
  fi
}

write_root_metadata

if [[ "$RUN_CORRECTNESS" == "1" ]]; then
  run_correctness
fi

if [[ "$RUN_SPEED" == "1" ]]; then
  run_speed
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  run_training
fi

finalize
