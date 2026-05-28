#!/usr/bin/env bash
set -euo pipefail

# Phase C: harder 16x16 semantic racing.
#
# Continues from the Phase B/lap-terminal baseline when present, then increases:
# - yaw change between gates
# - pitch/height variation
# - spacing between gates
# - post-gate reset frequency and lateral/yaw error
#
# This is intended to test reacquisition and turning, not just straight gates.

if [[ -z "${BASE:-}" ]]; then
  BASE="$(find runs/v0_minimal_active_vision_16x16_race_semantic_phaseB_lapterm_banked_seed44_1b \
    -type f -name '*.bin' 2>/dev/null | sort -V | tail -n 1 || true)"
fi
export BASE="${BASE:-}"

export BATCH_ID="${BATCH_ID:-v0_minimal_active_vision_16x16_race_semantic_phaseC_turns_from_phaseB_seed44_500m}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"

export TASK=7
export MAX_RINGS="${MAX_RINGS:-8}"
export TARGET_DIST="${TARGET_DIST:-5.0}"
export OOB_RADIUS="${OOB_RADIUS:-18.0}"
export RESET_POS_SCALE="${RESET_POS_SCALE:-0.08}"
export RESET_YAW_RANGE="${RESET_YAW_RANGE:-0.60}"
export RESET_VEL_MAX="${RESET_VEL_MAX:-0.0}"

export MINIMAL_VISION_ENABLED=1.0
export MINIMAL_VISION_ONLY="${MINIMAL_VISION_ONLY:-0.0}"
export MINIMAL_VISION_MASK_TARGET=1.0
export MINIMAL_VISION_SPAWN_VISIBLE_TARGET=1.0
export MINIMAL_VISION_WIDTH=16
export MINIMAL_VISION_HEIGHT=16
export MINIMAL_VISION_FOV="${MINIMAL_VISION_FOV:-2.0943951}"
export MINIMAL_VISION_VFOV="${MINIMAL_VISION_VFOV:-1.3962634}"
export MINIMAL_VISION_SIGMA="${MINIMAL_VISION_SIGMA:-0.22}"
export MINIMAL_VISION_DEPTH_GAIN="${MINIMAL_VISION_DEPTH_GAIN:-0.08}"
export MINIMAL_VISION_NOISE="${MINIMAL_VISION_NOISE:-0.02}"
export MINIMAL_VISION_DISTRACTORS="${MINIMAL_VISION_DISTRACTORS:-0.0}"

export RACE_COURSE_YAW_DELTA="${RACE_COURSE_YAW_DELTA:-0.70}"
export RACE_COURSE_PITCH_DELTA="${RACE_COURSE_PITCH_DELTA:-0.32}"
export RACE_COURSE_PITCH_LIMIT="${RACE_COURSE_PITCH_LIMIT:-0.45}"
export RACE_COURSE_SPACING_MIN="${RACE_COURSE_SPACING_MIN:-4.0}"
export RACE_COURSE_SPACING_MAX="${RACE_COURSE_SPACING_MAX:-6.0}"
export RACE_COURSE_DZ_MAX="${RACE_COURSE_DZ_MAX:-0.70}"

export RACE_RESET_START_PROB="${RACE_RESET_START_PROB:-0.60}"
export RACE_RESET_T_MIN="${RACE_RESET_T_MIN:-0.10}"
export RACE_RESET_T_MAX="${RACE_RESET_T_MAX:-0.70}"
export RACE_RESET_LATERAL="${RACE_RESET_LATERAL:-0.55}"
export RACE_RESET_YAW_ERROR_FRAC="${RACE_RESET_YAW_ERROR_FRAC:-0.18}"
export RACE_RESET_SPEED_MIN="${RACE_RESET_SPEED_MIN:-0.5}"
export RACE_RESET_SPEED_MAX="${RACE_RESET_SPEED_MAX:-2.2}"

export ALPHA_DIST="${ALPHA_DIST:-0.5}"
export ALPHA_HOVER="${ALPHA_HOVER:-0.0}"
export ALPHA_SHAPING="${ALPHA_SHAPING:-0.0}"
export ALPHA_OMEGA_XY="${ALPHA_OMEGA_XY:-0.004}"
export ALPHA_OMEGA_Z="${ALPHA_OMEGA_Z:-0.004}"
export ALPHA_OMEGA_Z_SQ="${ALPHA_OMEGA_Z_SQ:-0.006}"
export ALPHA_OMEGA_Z_MULT="${ALPHA_OMEGA_Z_MULT:-1.5}"
export ALPHA_ACTION_DELTA="${ALPHA_ACTION_DELTA:-0.01}"
export ALPHA_RESET_ACTION_DELTA="${ALPHA_RESET_ACTION_DELTA:-0.0}"

export ACTION_MODE="${ACTION_MODE:-1}"
export ACTION_SCALE="${ACTION_SCALE:-1.0}"
export NORMALIZED_THRUST_MAX="${NORMALIZED_THRUST_MAX:-0.80}"
export LEARNING_RATE="${LEARNING_RATE:-0.001}"
export MINIBATCH_SIZE="${MINIBATCH_SIZE:-16384}"
export HORIZON="${HORIZON:-64}"
export POLICY_HIDDEN_SIZE="${POLICY_HIDDEN_SIZE:-128}"
export TOTAL_AGENTS="${TOTAL_AGENTS:-8192}"
export NUM_DRONES="${NUM_DRONES:-8192}"

exec bash experiments/run_minimal_active_vision_ppo.sh
