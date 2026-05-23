#!/usr/bin/env bash
set -euo pipefail

# PPO-PAL pilot for the v0.14 legacy hover-trim baseline.
#
# This keeps the v0.14 action/observation/network contract intact so the
# baseline checkpoint can be used as a warm-start:
# - action_mode=0 / hover_trim
# - action_scale=0.7
# - OBS_SIZE=23
# - MinGRU policy with 3 layers
#
# PAL pieces in this pilot:
# - existing recurrent MinGRU as the latent dynamics adapter
# - probe-conditioned early action pulses, with dropout by episode
# - structured Pareto-anchor sampler via dr_profile_mix=3.x:
#     medium/hard replay, hard_small, capped, 3+1 mismatch, fast/slow motors
# - smoothness/reset penalties through env reward
# - no EPOpt tail overweighting by default

ENV_NAME="${ENV_NAME:-drone}"
BATCH_ID="${BATCH_ID:-v0_15_ppo_pal_probe_anchor_from_v014_seed44_500m}"
BATCH_ROOT="${BATCH_ROOT:-runs/$BATCH_ID}"
BASE="${BASE:-}"
BASE_LABEL="${BASE_LABEL:-v014_general_baseline}"
SEED="${SEED:-44}"
GPU_ID="${GPU_ID:-0}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
BUILD_FIRST="${BUILD_FIRST:-0}"

TOTAL_AGENTS="${TOTAL_AGENTS:-8192}"
NUM_BUFFERS="${NUM_BUFFERS:-1}"
NUM_THREADS="${NUM_THREADS:-32}"
NUM_DRONES="${NUM_DRONES:-8192}"

# Lower than the earlier ADR runs to reduce drift from v0.14.
LEARNING_RATE="${LEARNING_RATE:-0.000609395625}"
EPOPT_ALPHA="${EPOPT_ALPHA:-0.0}"
EPOPT_QUANTILE="${EPOPT_QUANTILE:-0.20}"
ACTION_SCALE="${ACTION_SCALE:-0.7}"
ACTION_MODE="${ACTION_MODE:-0}"
TARGET_DIST="${TARGET_DIST:-5.0}"
OOB_RADIUS="${OOB_RADIUS:-12.0}"
YAW_MULT="${YAW_MULT:-5.0}"
RESET_POS_SCALE="${RESET_POS_SCALE:-1.0}"
RESET_YAW_RANGE="${RESET_YAW_RANGE:-3.14159}"
RESET_VEL_MAX="${RESET_VEL_MAX:-0.2}"
ACTION_LATENCY="${ACTION_LATENCY:-0.01}"
SENSOR_NOISE="${SENSOR_NOISE:-0.01}"
ALPHA_ACTION_DELTA="${ALPHA_ACTION_DELTA:-0.005}"
ALPHA_RESET_ACTION_DELTA="${ALPHA_RESET_ACTION_DELTA:-0.010}"
RESET_ACTION_INTERVAL="${RESET_ACTION_INTERVAL:-32}"
NORMALIZED_THRUST_MAX="${NORMALIZED_THRUST_MAX:-0.85}"

# Probe-conditioned latent adaptation. The probe is injected by the env before
# motor dynamics, only for the first PAL_PROBE_STEPS control steps in sampled
# episodes. It is intentionally tiny relative to [-1, 1] hover-trim action.
PAL_PROBE_PROB="${PAL_PROBE_PROB:-0.50}"
PAL_PROBE_STEPS="${PAL_PROBE_STEPS:-16}"
PAL_PROBE_AMP="${PAL_PROBE_AMP:-0.030}"

# Structured legacy profile: guard/core plus hidden-dynamics corners.
DR_PROFILE_MIX="${DR_PROFILE_MIX:-3.0}"
DR_USABLE_T2W_MIN="${DR_USABLE_T2W_MIN:-2.00}"
DR_USABLE_T2W_MAX="${DR_USABLE_T2W_MAX:-4.20}"
DR_MASS_MIN="${DR_MASS_MIN:-0.75}"
DR_MASS_MAX="${DR_MASS_MAX:-1.25}"
DR_INERTIA_MIN="${DR_INERTIA_MIN:-0.45}"
DR_INERTIA_MAX="${DR_INERTIA_MAX:-1.65}"
DR_MOTOR_THRUST_MIN="${DR_MOTOR_THRUST_MIN:-0.78}"
DR_MOTOR_THRUST_MAX="${DR_MOTOR_THRUST_MAX:-1.18}"
DR_MOTOR_TAU_MIN="${DR_MOTOR_TAU_MIN:-0.05}"
DR_MOTOR_TAU_MAX="${DR_MOTOR_TAU_MAX:-0.28}"
DR_YAW_TORQUE_MIN="${DR_YAW_TORQUE_MIN:-0.70}"
DR_YAW_TORQUE_MAX="${DR_YAW_TORQUE_MAX:-1.30}"
DR_COM_XY="${DR_COM_XY:-0.030}"
DR_COM_Z="${DR_COM_Z:-0.018}"
DR_LINEAR_DRAG_MIN="${DR_LINEAR_DRAG_MIN:-0.25}"
DR_LINEAR_DRAG_MAX="${DR_LINEAR_DRAG_MAX:-2.00}"
DR_ANGULAR_DAMPING_MIN="${DR_ANGULAR_DAMPING_MIN:-0.50}"
DR_ANGULAR_DAMPING_MAX="${DR_ANGULAR_DAMPING_MAX:-2.00}"

ADR_ENABLED="${ADR_ENABLED:-0.0}"
ADR_PROBE_PROB="${ADR_PROBE_PROB:-0.0}"
ADR_EVAL_EPISODES="${ADR_EVAL_EPISODES:-240}"
ADR_SUCCESS_THRESHOLD="${ADR_SUCCESS_THRESHOLD:-0.94}"
ADR_CONTRACT_THRESHOLD="${ADR_CONTRACT_THRESHOLD:-0.55}"
ADR_STEP="${ADR_STEP:-0.01}"

if [[ -z "$BASE" ]]; then
  candidates=(
    "baselines/general/policy.bin"
    "baselines/v0_14_dr_medium_robust_seed44_scale0.7_reset32_latest.bin"
    "artifacts/v0.14-dr-medium-robust-baseline/v0_14_dr_medium_robust_seed44_scale0.7_reset32_latest.bin"
    "artifacts/legacy_policy_extracts/v0_14_dr_medium_robust_seed44_scale0.7_reset32_latest.bin"
    "base_checkpoints/v0_14_general_baseline_policy.bin"
    "/workspace/matrice4d-pufferlib-v0/base_checkpoints/v0_14_general_baseline_policy.bin"
    "/workspace/matrice4d-pufferlib-v0/baselines/general/policy.bin"
    "/Users/alan/matrice4d_experiments/baselines/general/policy.bin"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      BASE="$candidate"
      break
    fi
  done
fi

if [[ -z "$BASE" || ! -f "$BASE" ]]; then
  cat >&2 <<'EOF'
Missing v0.14 base checkpoint.

Set BASE to the v0.14 general baseline checkpoint, for example:

BASE=/workspace/matrice4d-pufferlib-v0/baselines/general/policy.bin \
  bash experiments/run_v0_15_ppo_pal_from_v014.sh
EOF
  exit 2
fi

mkdir -p "$BATCH_ROOT/checkpoints" "$BATCH_ROOT/logs" artifacts
git rev-parse HEAD > "$BATCH_ROOT/git_commit.txt" 2>/dev/null || true
git status --short > "$BATCH_ROOT/git_status.txt" 2>/dev/null || true
git diff > "$BATCH_ROOT/git_diff.patch" 2>/dev/null || true

if [[ "$BUILD_FIRST" == "1" ]]; then
  bash build.sh "$ENV_NAME" 2>&1 | tee "$BATCH_ROOT/build_stdout.txt"
fi

{
  echo "experiment=v0_15_ppo_pal_probe_anchor_from_v014"
  echo "base_checkpoint=$BASE"
  echo "base_label=$BASE_LABEL"
  echo "seed=$SEED"
  echo "total_timesteps=$TOTAL_TIMESTEPS"
  echo "learning_rate=$LEARNING_RATE"
  echo "epopt_alpha=$EPOPT_ALPHA"
  echo "epopt_quantile=$EPOPT_QUANTILE"
  echo "action_mode=$ACTION_MODE"
  echo "action_scale=$ACTION_SCALE"
  echo "pal_probe_prob=$PAL_PROBE_PROB"
  echo "pal_probe_steps=$PAL_PROBE_STEPS"
  echo "pal_probe_amp=$PAL_PROBE_AMP"
  echo "alpha_action_delta=$ALPHA_ACTION_DELTA"
  echo "alpha_reset_action_delta=$ALPHA_RESET_ACTION_DELTA"
  echo "reset_action_interval=$RESET_ACTION_INTERVAL"
  echo "dr_profile_mix=$DR_PROFILE_MIX"
  echo "adr_enabled=$ADR_ENABLED"
  echo "action_latency=$ACTION_LATENCY"
  echo "sensor_noise=$SENSOR_NOISE"
  echo "dr_usable_t2w_min=$DR_USABLE_T2W_MIN"
  echo "dr_usable_t2w_max=$DR_USABLE_T2W_MAX"
  echo "dr_mass_min=$DR_MASS_MIN"
  echo "dr_mass_max=$DR_MASS_MAX"
  echo "dr_inertia_min=$DR_INERTIA_MIN"
  echo "dr_inertia_max=$DR_INERTIA_MAX"
  echo "dr_motor_thrust_min=$DR_MOTOR_THRUST_MIN"
  echo "dr_motor_thrust_max=$DR_MOTOR_THRUST_MAX"
  echo "dr_motor_tau_min=$DR_MOTOR_TAU_MIN"
  echo "dr_motor_tau_max=$DR_MOTOR_TAU_MAX"
  echo "dr_com_xy=$DR_COM_XY"
  echo "dr_com_z=$DR_COM_Z"
} > "$BATCH_ROOT/run_metadata.env"

cmd=(
  python -m pufferlib.pufferl train "$ENV_NAME"
  --tag "$BATCH_ID"
  --load-model-path "$BASE"
  --checkpoint-dir "$BATCH_ROOT/checkpoints"
  --log-dir "$BATCH_ROOT/logs"
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --seed "$SEED"
  --train.seed "$SEED"
  --train.total-timesteps "$TOTAL_TIMESTEPS"
  --train.learning-rate "$LEARNING_RATE"
  --train.epopt-alpha "$EPOPT_ALPHA"
  --train.epopt-quantile "$EPOPT_QUANTILE"
  --vec.total-agents "$TOTAL_AGENTS"
  --vec.num-buffers "$NUM_BUFFERS"
  --vec.num-threads "$NUM_THREADS"
  --env.num-drones "$NUM_DRONES"
  --env.domain-randomization 1.0
  --env.dr-authority-gated 1.0
  --env.dr-profile-mix "$DR_PROFILE_MIX"
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
  --env.adr-enabled "$ADR_ENABLED"
  --env.adr-mode 1.0
  --env.adr-probe-prob "$ADR_PROBE_PROB"
  --env.adr-eval-episodes "$ADR_EVAL_EPISODES"
  --env.adr-success-threshold "$ADR_SUCCESS_THRESHOLD"
  --env.adr-contract-threshold "$ADR_CONTRACT_THRESHOLD"
  --env.adr-step "$ADR_STEP"
  --env.pal-probe-prob "$PAL_PROBE_PROB"
  --env.pal-probe-steps "$PAL_PROBE_STEPS"
  --env.pal-probe-amp "$PAL_PROBE_AMP"
  --env.action-scale "$ACTION_SCALE"
  --env.action-mode "$ACTION_MODE"
  --env.normalized-thrust-min 0.0
  --env.normalized-thrust-max "$NORMALIZED_THRUST_MAX"
  --env.hover-target-dist "$TARGET_DIST"
  --env.oob-radius "$OOB_RADIUS"
  --env.alpha-omega-z-mult "$YAW_MULT"
  --env.alpha-action-delta "$ALPHA_ACTION_DELTA"
  --env.alpha-reset-action-delta "$ALPHA_RESET_ACTION_DELTA"
  --env.reset-action-interval "$RESET_ACTION_INTERVAL"
  --env.reset-pos-scale "$RESET_POS_SCALE"
  --env.reset-yaw-range "$RESET_YAW_RANGE"
  --env.reset-vel-max "$RESET_VEL_MAX"
  --env.action-latency "$ACTION_LATENCY"
  --env.sensor-noise "$SENSOR_NOISE"
  --policy.num-layers 3
)

printf '%q ' CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" > "$BATCH_ROOT/command.sh"
printf '\n' >> "$BATCH_ROOT/command.sh"
chmod +x "$BATCH_ROOT/command.sh"

echo "### Running PPO-PAL pilot from v0.14: $BATCH_ID"
echo "### Base: $BASE"
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
