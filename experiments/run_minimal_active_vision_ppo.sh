#!/usr/bin/env bash
set -euo pipefail

# Minimal Active Vision PPO pilot.
#
# This trains from scratch with the existing 4-motor low-level action contract,
# but removes the perfect target vector from the observation. The policy keeps
# proprioception for stabilization and gets target direction only through a
# 1x3 RGB retina appended to the observation.

ENV_NAME="${ENV_NAME:-drone}"
BATCH_ID="${BATCH_ID:-v0_minimal_active_vision_3x1_targetmask_seed44_100m}"
BATCH_ROOT="${BATCH_ROOT:-runs/$BATCH_ID}"
SEED="${SEED:-44}"
GPU_ID="${GPU_ID:-0}"
BASE="${BASE:-}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
BUILD_FIRST="${BUILD_FIRST:-1}"

TOTAL_AGENTS="${TOTAL_AGENTS:-8192}"
NUM_BUFFERS="${NUM_BUFFERS:-1}"
NUM_THREADS="${NUM_THREADS:-32}"
NUM_DRONES="${NUM_DRONES:-8192}"

LEARNING_RATE="${LEARNING_RATE:-0.003}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-16384}"
HORIZON="${HORIZON:-64}"

TASK="${TASK:-1}"
MAX_RINGS="${MAX_RINGS:-10}"
ACTION_MODE="${ACTION_MODE:-1}"
ACTION_SCALE="${ACTION_SCALE:-1.0}"
NORMALIZED_THRUST_MAX="${NORMALIZED_THRUST_MAX:-0.85}"

TARGET_DIST="${TARGET_DIST:-5.0}"
OOB_RADIUS="${OOB_RADIUS:-18.0}"
RESET_POS_SCALE="${RESET_POS_SCALE:-0.20}"
RESET_YAW_RANGE="${RESET_YAW_RANGE:-0.60}"
RESET_VEL_MAX="${RESET_VEL_MAX:-0.0}"

ALPHA_DIST="${ALPHA_DIST:-1.0}"
ALPHA_HOVER="${ALPHA_HOVER:-0.02}"
ALPHA_SHAPING="${ALPHA_SHAPING:-1.0}"
ALPHA_OMEGA_XY="${ALPHA_OMEGA_XY:-0.0015}"
ALPHA_OMEGA_Z="${ALPHA_OMEGA_Z:-0.0015}"
ALPHA_OMEGA_Z_SQ="${ALPHA_OMEGA_Z_SQ:-0.0025}"
ALPHA_OMEGA_Z_MULT="${ALPHA_OMEGA_Z_MULT:-1.0}"
ALPHA_ACTION_DELTA="${ALPHA_ACTION_DELTA:-0.002}"
ALPHA_RESET_ACTION_DELTA="${ALPHA_RESET_ACTION_DELTA:-0.0}"

MINIMAL_VISION_ENABLED="${MINIMAL_VISION_ENABLED:-1.0}"
MINIMAL_VISION_ONLY="${MINIMAL_VISION_ONLY:-0.0}"
MINIMAL_VISION_MASK_TARGET="${MINIMAL_VISION_MASK_TARGET:-1.0}"
MINIMAL_VISION_SPAWN_VISIBLE_TARGET="${MINIMAL_VISION_SPAWN_VISIBLE_TARGET:-1.0}"
MINIMAL_VISION_FOV="${MINIMAL_VISION_FOV:-2.0943951}"
MINIMAL_VISION_VFOV="${MINIMAL_VISION_VFOV:-1.3962634}"
MINIMAL_VISION_SIGMA="${MINIMAL_VISION_SIGMA:-0.45}"
MINIMAL_VISION_DEPTH_GAIN="${MINIMAL_VISION_DEPTH_GAIN:-0.08}"
MINIMAL_VISION_NOISE="${MINIMAL_VISION_NOISE:-0.02}"
MINIMAL_VISION_DISTRACTORS="${MINIMAL_VISION_DISTRACTORS:-0.0}"
MINIMAL_VISION_WIDTH="${MINIMAL_VISION_WIDTH:-3}"
MINIMAL_VISION_HEIGHT="${MINIMAL_VISION_HEIGHT:-1}"
RACE_TRACK_MODE="${RACE_TRACK_MODE:-0.0}"
RACE_COURSE_YAW_DELTA="${RACE_COURSE_YAW_DELTA:-0.0}"
RACE_COURSE_PITCH_DELTA="${RACE_COURSE_PITCH_DELTA:-0.0}"
RACE_COURSE_PITCH_LIMIT="${RACE_COURSE_PITCH_LIMIT:-0.0}"
RACE_COURSE_SPACING_MIN="${RACE_COURSE_SPACING_MIN:-0.0}"
RACE_COURSE_SPACING_MAX="${RACE_COURSE_SPACING_MAX:-0.0}"
RACE_COURSE_DZ_MAX="${RACE_COURSE_DZ_MAX:-0.0}"
RACE_RESET_START_PROB="${RACE_RESET_START_PROB:-0.0}"
RACE_RESET_T_MIN="${RACE_RESET_T_MIN:-0.0}"
RACE_RESET_T_MAX="${RACE_RESET_T_MAX:-0.0}"
RACE_RESET_LATERAL="${RACE_RESET_LATERAL:-0.0}"
RACE_RESET_YAW_ERROR_FRAC="${RACE_RESET_YAW_ERROR_FRAC:-0.0}"
RACE_RESET_SPEED_MIN="${RACE_RESET_SPEED_MIN:-0.0}"
RACE_RESET_SPEED_MAX="${RACE_RESET_SPEED_MAX:-0.0}"
RACE_SEGMENT_MODE="${RACE_SEGMENT_MODE:-0.0}"
RACE_ISB_ENABLED="${RACE_ISB_ENABLED:-0.0}"
RACE_ISB_PROB="${RACE_ISB_PROB:-0.0}"
RACE_ISB_MARGIN="${RACE_ISB_MARGIN:-0.8}"
RACE_ISB_POS_XY="${RACE_ISB_POS_XY:-0.45}"
RACE_ISB_Z="${RACE_ISB_Z:-0.25}"
RACE_ISB_ANGLE="${RACE_ISB_ANGLE:-0.18}"
RACE_ISB_VEL="${RACE_ISB_VEL:-0.60}"
RACE_ISB_OMEGA="${RACE_ISB_OMEGA:-0.60}"
RACE_HARD_GATE_IDX="${RACE_HARD_GATE_IDX:--1.0}"
RACE_HARD_GATE_PROB="${RACE_HARD_GATE_PROB:-0.0}"
POLICY_HIDDEN_SIZE="${POLICY_HIDDEN_SIZE:-128}"

mkdir -p "$BATCH_ROOT/checkpoints" "$BATCH_ROOT/logs" artifacts
git rev-parse HEAD > "$BATCH_ROOT/git_commit.txt" 2>/dev/null || true
git status --short > "$BATCH_ROOT/git_status.txt" 2>/dev/null || true
git diff > "$BATCH_ROOT/git_diff.patch" 2>/dev/null || true

if [[ "$BUILD_FIRST" == "1" ]]; then
  DRONE_MINIMAL_VISION_WIDTH="$MINIMAL_VISION_WIDTH" \
  DRONE_MINIMAL_VISION_HEIGHT="$MINIMAL_VISION_HEIGHT" \
    bash build.sh "$ENV_NAME" 2>&1 | tee "$BATCH_ROOT/build_stdout.txt"
fi

{
  echo "experiment=minimal_active_vision_ppo"
  echo "seed=$SEED"
  echo "base=$BASE"
  echo "total_timesteps=$TOTAL_TIMESTEPS"
  echo "learning_rate=$LEARNING_RATE"
  echo "task=$TASK"
  echo "max_rings=$MAX_RINGS"
  echo "action_mode=$ACTION_MODE"
  echo "normalized_thrust_max=$NORMALIZED_THRUST_MAX"
  echo "target_dist=$TARGET_DIST"
  echo "oob_radius=$OOB_RADIUS"
  echo "reset_pos_scale=$RESET_POS_SCALE"
  echo "reset_yaw_range=$RESET_YAW_RANGE"
  echo "reset_vel_max=$RESET_VEL_MAX"
  echo "alpha_dist=$ALPHA_DIST"
  echo "alpha_hover=$ALPHA_HOVER"
  echo "alpha_shaping=$ALPHA_SHAPING"
  echo "alpha_omega_xy=$ALPHA_OMEGA_XY"
  echo "alpha_omega_z=$ALPHA_OMEGA_Z"
  echo "alpha_omega_z_sq=$ALPHA_OMEGA_Z_SQ"
  echo "alpha_omega_z_mult=$ALPHA_OMEGA_Z_MULT"
  echo "alpha_action_delta=$ALPHA_ACTION_DELTA"
  echo "alpha_reset_action_delta=$ALPHA_RESET_ACTION_DELTA"
  echo "minimal_vision_enabled=$MINIMAL_VISION_ENABLED"
  echo "minimal_vision_only=$MINIMAL_VISION_ONLY"
  echo "minimal_vision_mask_target=$MINIMAL_VISION_MASK_TARGET"
  echo "minimal_vision_spawn_visible_target=$MINIMAL_VISION_SPAWN_VISIBLE_TARGET"
  echo "minimal_vision_fov=$MINIMAL_VISION_FOV"
  echo "minimal_vision_vfov=$MINIMAL_VISION_VFOV"
  echo "minimal_vision_sigma=$MINIMAL_VISION_SIGMA"
  echo "minimal_vision_depth_gain=$MINIMAL_VISION_DEPTH_GAIN"
  echo "minimal_vision_noise=$MINIMAL_VISION_NOISE"
  echo "minimal_vision_distractors=$MINIMAL_VISION_DISTRACTORS"
  echo "minimal_vision_width=$MINIMAL_VISION_WIDTH"
  echo "minimal_vision_height=$MINIMAL_VISION_HEIGHT"
  echo "race_track_mode=$RACE_TRACK_MODE"
  echo "race_segment_mode=$RACE_SEGMENT_MODE"
  echo "race_isb_enabled=$RACE_ISB_ENABLED"
  echo "race_isb_prob=$RACE_ISB_PROB"
  echo "race_isb_margin=$RACE_ISB_MARGIN"
  echo "race_isb_pos_xy=$RACE_ISB_POS_XY"
  echo "race_isb_z=$RACE_ISB_Z"
  echo "race_isb_angle=$RACE_ISB_ANGLE"
  echo "race_isb_vel=$RACE_ISB_VEL"
  echo "race_isb_omega=$RACE_ISB_OMEGA"
  echo "race_hard_gate_idx=$RACE_HARD_GATE_IDX"
  echo "race_hard_gate_prob=$RACE_HARD_GATE_PROB"
  echo "race_course_yaw_delta=$RACE_COURSE_YAW_DELTA"
  echo "race_course_pitch_delta=$RACE_COURSE_PITCH_DELTA"
  echo "race_course_pitch_limit=$RACE_COURSE_PITCH_LIMIT"
  echo "race_course_spacing_min=$RACE_COURSE_SPACING_MIN"
  echo "race_course_spacing_max=$RACE_COURSE_SPACING_MAX"
  echo "race_course_dz_max=$RACE_COURSE_DZ_MAX"
  echo "race_reset_start_prob=$RACE_RESET_START_PROB"
  echo "race_reset_t_min=$RACE_RESET_T_MIN"
  echo "race_reset_t_max=$RACE_RESET_T_MAX"
  echo "race_reset_lateral=$RACE_RESET_LATERAL"
  echo "race_reset_yaw_error_frac=$RACE_RESET_YAW_ERROR_FRAC"
  echo "race_reset_speed_min=$RACE_RESET_SPEED_MIN"
  echo "race_reset_speed_max=$RACE_RESET_SPEED_MAX"
  echo "policy_hidden_size=$POLICY_HIDDEN_SIZE"
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
  --train.minibatch-size "$MINIBATCH_SIZE"
  --train.horizon "$HORIZON"
  --train.ent-coef 0.001
  --train.clip-coef 0.12
  --train.vf-coef 2.0
  --train.gamma 0.99
  --train.gae-lambda 0.90
  --vec.total-agents "$TOTAL_AGENTS"
  --vec.num-buffers "$NUM_BUFFERS"
  --vec.num-threads "$NUM_THREADS"
  --env.num-drones "$NUM_DRONES"
  --env.task "$TASK"
  --env.max-rings "$MAX_RINGS"
  --env.action-mode "$ACTION_MODE"
  --env.action-scale "$ACTION_SCALE"
  --env.normalized-thrust-min 0.0
  --env.normalized-thrust-max "$NORMALIZED_THRUST_MAX"
  --env.hover-target-dist "$TARGET_DIST"
  --env.oob-radius "$OOB_RADIUS"
  --env.reset-pos-scale "$RESET_POS_SCALE"
  --env.reset-yaw-range "$RESET_YAW_RANGE"
  --env.reset-vel-max "$RESET_VEL_MAX"
  --env.alpha-dist "$ALPHA_DIST"
  --env.alpha-hover "$ALPHA_HOVER"
  --env.alpha-shaping "$ALPHA_SHAPING"
  --env.alpha-omega-xy "$ALPHA_OMEGA_XY"
  --env.alpha-omega-z "$ALPHA_OMEGA_Z"
  --env.alpha-omega-z-sq "$ALPHA_OMEGA_Z_SQ"
  --env.alpha-omega-z-mult "$ALPHA_OMEGA_Z_MULT"
  --env.alpha-action-delta "$ALPHA_ACTION_DELTA"
  --env.alpha-reset-action-delta "$ALPHA_RESET_ACTION_DELTA"
  --env.minimal-vision-enabled "$MINIMAL_VISION_ENABLED"
  --env.minimal-vision-only "$MINIMAL_VISION_ONLY"
  --env.minimal-vision-mask-target "$MINIMAL_VISION_MASK_TARGET"
  --env.minimal-vision-spawn-visible-target "$MINIMAL_VISION_SPAWN_VISIBLE_TARGET"
  --env.minimal-vision-fov "$MINIMAL_VISION_FOV"
  --env.minimal-vision-vfov "$MINIMAL_VISION_VFOV"
  --env.minimal-vision-sigma "$MINIMAL_VISION_SIGMA"
  --env.minimal-vision-depth-gain "$MINIMAL_VISION_DEPTH_GAIN"
  --env.minimal-vision-noise "$MINIMAL_VISION_NOISE"
  --env.minimal-vision-distractors "$MINIMAL_VISION_DISTRACTORS"
  --env.race-track-mode "$RACE_TRACK_MODE"
  --env.race-segment-mode "$RACE_SEGMENT_MODE"
  --env.race-isb-enabled "$RACE_ISB_ENABLED"
  --env.race-isb-prob "$RACE_ISB_PROB"
  --env.race-isb-margin "$RACE_ISB_MARGIN"
  --env.race-isb-pos-xy "$RACE_ISB_POS_XY"
  --env.race-isb-z "$RACE_ISB_Z"
  --env.race-isb-angle "$RACE_ISB_ANGLE"
  --env.race-isb-vel "$RACE_ISB_VEL"
  --env.race-isb-omega "$RACE_ISB_OMEGA"
  --env.race-hard-gate-idx "$RACE_HARD_GATE_IDX"
  --env.race-hard-gate-prob "$RACE_HARD_GATE_PROB"
  --env.race-course-yaw-delta "$RACE_COURSE_YAW_DELTA"
  --env.race-course-pitch-delta "$RACE_COURSE_PITCH_DELTA"
  --env.race-course-pitch-limit "$RACE_COURSE_PITCH_LIMIT"
  --env.race-course-spacing-min "$RACE_COURSE_SPACING_MIN"
  --env.race-course-spacing-max "$RACE_COURSE_SPACING_MAX"
  --env.race-course-dz-max "$RACE_COURSE_DZ_MAX"
  --env.race-reset-start-prob "$RACE_RESET_START_PROB"
  --env.race-reset-t-min "$RACE_RESET_T_MIN"
  --env.race-reset-t-max "$RACE_RESET_T_MAX"
  --env.race-reset-lateral "$RACE_RESET_LATERAL"
  --env.race-reset-yaw-error-frac "$RACE_RESET_YAW_ERROR_FRAC"
  --env.race-reset-speed-min "$RACE_RESET_SPEED_MIN"
  --env.race-reset-speed-max "$RACE_RESET_SPEED_MAX"
  --policy.hidden-size "$POLICY_HIDDEN_SIZE"
  --policy.num-layers 3
)

if [[ -n "$BASE" ]]; then
  if [[ ! -f "$BASE" ]]; then
    echo "Missing base checkpoint: $BASE" >&2
    exit 2
  fi
  cmd+=(--load-model-path "$BASE")
fi

printf '%q ' CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" > "$BATCH_ROOT/command.sh"
printf '\n' >> "$BATCH_ROOT/command.sh"
chmod +x "$BATCH_ROOT/command.sh"

echo "### Running Minimal Active Vision PPO: $BATCH_ID"
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
