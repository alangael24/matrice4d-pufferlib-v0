#!/usr/bin/env bash
set -euo pipefail

# First 3x1 raw-camera race experiment.
#
# This branch changes the drone observation contract:
#   23 base proprio/target floats + 9 raw camera floats = 32 obs.
#
# Do not warm-start from old 23-observation checkpoints. BASE is only for a
# future camera-compatible checkpoint.

ENV_NAME="${ENV_NAME:-drone}"
BATCH_ID="${BATCH_ID:-v0_camera_3x1_race_from_scratch_100m}"
BATCH_ROOT="${BATCH_ROOT:-runs/$BATCH_ID}"
BASE="${BASE:-}"
SEED="${SEED:-44}"
GPU_ID="${GPU_ID:-0}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
BUILD_FIRST="${BUILD_FIRST:-0}"

TOTAL_AGENTS="${TOTAL_AGENTS:-8192}"
NUM_BUFFERS="${NUM_BUFFERS:-1}"
NUM_THREADS="${NUM_THREADS:-32}"
NUM_DRONES="${NUM_DRONES:-8192}"

LEARNING_RATE="${LEARNING_RATE:-0.00075}"
ACTION_SCALE="${ACTION_SCALE:-0.7}"
ACTION_MODE="${ACTION_MODE:-0}"
OOB_RADIUS="${OOB_RADIUS:-16.0}"
YAW_MULT="${YAW_MULT:-5.0}"
SENSOR_NOISE="${SENSOR_NOISE:-0.005}"
ACTION_LATENCY="${ACTION_LATENCY:-0.01}"
ALPHA_ACTION_DELTA="${ALPHA_ACTION_DELTA:-0.002}"
ALPHA_RESET_ACTION_DELTA="${ALPHA_RESET_ACTION_DELTA:-0.004}"
RESET_ACTION_INTERVAL="${RESET_ACTION_INTERVAL:-32}"

CAMERA_FOV_X="${CAMERA_FOV_X:-120.0}"
CAMERA_FOV_Y="${CAMERA_FOV_Y:-80.0}"
CAMERA_GATE_GAIN="${CAMERA_GATE_GAIN:-1.0}"
CAMERA_BG="${CAMERA_BG:-0.02}"
CAMERA_NOISE="${CAMERA_NOISE:-0.02}"

RACE_GATE_SPACING="${RACE_GATE_SPACING:-10.0}"
RACE_LATERAL_RANGE="${RACE_LATERAL_RANGE:-4.0}"
RACE_VERTICAL_RANGE="${RACE_VERTICAL_RANGE:-2.0}"
RACE_SPAWN_DIST="${RACE_SPAWN_DIST:-8.0}"
RACE_SPAWN_JITTER="${RACE_SPAWN_JITTER:-1.0}"
RACE_GATE_REWARD="${RACE_GATE_REWARD:-5.0}"
RACE_GATE_HIT_PENALTY="${RACE_GATE_HIT_PENALTY:-2.0}"
RACE_PROGRESS_SCALE="${RACE_PROGRESS_SCALE:-0.10}"

DR_ENABLED="${DR_ENABLED:-1.0}"
DR_MASS="${DR_MASS:-0.08}"
DR_INERTIA="${DR_INERTIA:-0.15}"
DR_K_THRUST="${DR_K_THRUST:-0.15}"
DR_LINEAR_DRAG="${DR_LINEAR_DRAG:-0.25}"
DR_YAW_DRAG="${DR_YAW_DRAG:-0.20}"
DR_MOTOR_LAG="${DR_MOTOR_LAG:-0.10}"
DR_COM_XY="${DR_COM_XY:-0.010}"
DR_COM_Z="${DR_COM_Z:-0.006}"

mkdir -p "$BATCH_ROOT/checkpoints" "$BATCH_ROOT/logs" artifacts
git rev-parse HEAD > "$BATCH_ROOT/git_commit.txt" 2>/dev/null || true
git status --short > "$BATCH_ROOT/git_status.txt" 2>/dev/null || true
git diff > "$BATCH_ROOT/git_diff.patch" 2>/dev/null || true

if [[ "$BUILD_FIRST" == "1" ]]; then
  bash build.sh "$ENV_NAME" 2>&1 | tee "$BATCH_ROOT/build_stdout.txt"
fi

{
  echo "experiment=v0_camera_3x1_race"
  echo "obs_contract=23_base_plus_9_camera_rgb"
  echo "base_checkpoint=$BASE"
  echo "seed=$SEED"
  echo "total_timesteps=$TOTAL_TIMESTEPS"
  echo "learning_rate=$LEARNING_RATE"
  echo "camera_fov_x=$CAMERA_FOV_X"
  echo "camera_fov_y=$CAMERA_FOV_Y"
  echo "race_gate_spacing=$RACE_GATE_SPACING"
  echo "race_lateral_range=$RACE_LATERAL_RANGE"
  echo "race_vertical_range=$RACE_VERTICAL_RANGE"
  echo "domain_randomization=$DR_ENABLED"
} > "$BATCH_ROOT/run_metadata.env"

cmd=(
  python -m pufferlib.pufferl train "$ENV_NAME"
  --tag "$BATCH_ID"
  --checkpoint-dir "$BATCH_ROOT/checkpoints"
  --log-dir "$BATCH_ROOT/logs"
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --seed "$SEED"
  --train.seed "$SEED"
  --train.total-timesteps "$TOTAL_TIMESTEPS"
  --train.learning-rate "$LEARNING_RATE"
  --vec.total-agents "$TOTAL_AGENTS"
  --vec.num-buffers "$NUM_BUFFERS"
  --vec.num-threads "$NUM_THREADS"
  --env.task 7
  --env.num-drones "$NUM_DRONES"
  --env.max-rings 10
  --env.camera-3x1-enabled 1.0
  --env.camera-fov-x "$CAMERA_FOV_X"
  --env.camera-fov-y "$CAMERA_FOV_Y"
  --env.camera-gate-gain "$CAMERA_GATE_GAIN"
  --env.camera-bg "$CAMERA_BG"
  --env.camera-noise "$CAMERA_NOISE"
  --env.race-gate-spacing "$RACE_GATE_SPACING"
  --env.race-lateral-range "$RACE_LATERAL_RANGE"
  --env.race-vertical-range "$RACE_VERTICAL_RANGE"
  --env.race-spawn-dist "$RACE_SPAWN_DIST"
  --env.race-spawn-jitter "$RACE_SPAWN_JITTER"
  --env.race-gate-reward "$RACE_GATE_REWARD"
  --env.race-gate-hit-penalty "$RACE_GATE_HIT_PENALTY"
  --env.race-progress-scale "$RACE_PROGRESS_SCALE"
  --env.oob-radius "$OOB_RADIUS"
  --env.alpha-dist 0.10
  --env.alpha-hover 0.0
  --env.alpha-shaping 0.0
  --env.alpha-omega-z-mult "$YAW_MULT"
  --env.alpha-action-delta "$ALPHA_ACTION_DELTA"
  --env.alpha-reset-action-delta "$ALPHA_RESET_ACTION_DELTA"
  --env.reset-action-interval "$RESET_ACTION_INTERVAL"
  --env.action-scale "$ACTION_SCALE"
  --env.action-mode "$ACTION_MODE"
  --env.action-latency "$ACTION_LATENCY"
  --env.sensor-noise "$SENSOR_NOISE"
  --env.domain-randomization "$DR_ENABLED"
  --env.dr-mass "$DR_MASS"
  --env.dr-inertia "$DR_INERTIA"
  --env.dr-k-thrust "$DR_K_THRUST"
  --env.dr-linear-drag "$DR_LINEAR_DRAG"
  --env.dr-yaw-drag "$DR_YAW_DRAG"
  --env.dr-motor-lag "$DR_MOTOR_LAG"
  --env.dr-com-xy "$DR_COM_XY"
  --env.dr-com-z "$DR_COM_Z"
  --policy.num-layers 3
)

if [[ -n "$BASE" ]]; then
  if [[ ! -f "$BASE" ]]; then
    echo "BASE was set but does not exist: $BASE" >&2
    exit 2
  fi
  cmd+=(--load-model-path "$BASE")
fi

printf '%q ' CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" > "$BATCH_ROOT/command.sh"
printf '\n' >> "$BATCH_ROOT/command.sh"
chmod +x "$BATCH_ROOT/command.sh"

echo "### Running 3x1 camera race: $BATCH_ID"
echo "### Logs: $BATCH_ROOT"
set +e
CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" 2>&1 | tee "$BATCH_ROOT/stdout.txt"
exit_code="${PIPESTATUS[0]}"
set -e
echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BATCH_ROOT/run_metadata.env"
echo "exit_code=$exit_code" >> "$BATCH_ROOT/run_metadata.env"

latest="$(find "$BATCH_ROOT/checkpoints" -type f -name '*.bin' | sort -V | tail -n 1 || true)"
if [[ -n "$latest" && -f "$latest" ]]; then
  package_dir="artifacts/downloads/$BATCH_ID"
  mkdir -p "$package_dir"
  cp "$latest" "$package_dir/"
  find "$BATCH_ROOT/checkpoints" -type f -name '*.bin' | sort -V | tail -n 5 | while read -r ckpt; do
    cp "$ckpt" "$package_dir/"
  done
  cp -f "$BATCH_ROOT"/run_metadata.env "$package_dir/" 2>/dev/null || true
  cp -f "$BATCH_ROOT"/command.sh "$package_dir/" 2>/dev/null || true
  cp -f "$BATCH_ROOT"/git_commit.txt "$package_dir/" 2>/dev/null || true
  cp -f "$BATCH_ROOT"/git_status.txt "$package_dir/" 2>/dev/null || true
  cp -f "$BATCH_ROOT"/git_diff.patch "$package_dir/" 2>/dev/null || true
  find "$BATCH_ROOT/logs" -maxdepth 3 -type f -name '*.json' -exec cp {} "$package_dir/" \; 2>/dev/null || true
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$package_dir"/* > "$package_dir/sha256.txt"
  else
    shasum -a 256 "$package_dir"/* > "$package_dir/sha256.txt"
  fi
  tar -czf "artifacts/${BATCH_ID}_minimal.tgz" -C artifacts/downloads "$BATCH_ID"
  echo "artifact=artifacts/${BATCH_ID}_minimal.tgz" >> "$BATCH_ROOT/run_metadata.env"
  echo "latest_checkpoint=$latest" >> "$BATCH_ROOT/run_metadata.env"
  echo "### Artifact: artifacts/${BATCH_ID}_minimal.tgz"
fi

exit "$exit_code"
