#!/usr/bin/env bash
set -euo pipefail

# Phase D: Swift-like 16x16 semantic racing.
#
# Uses a seven-gate fixed-layout family inspired by the Swift race track:
# long diagonals, large yaw changes, height changes, and a short vertical
# gate-to-gate transition. RACE_TRACK_MODE=2 randomizes global yaw, mirror,
# scale, and small translation each episode so this is not one memorized map.

if [[ -z "${BASE:-}" ]]; then
  for candidate_root in \
    runs/v0_minimal_active_vision_16x16_race_phaseC_turns_from_phaseB_seed44_500m \
    runs/v0_minimal_active_vision_16x16_race_semantic_phaseC_turns_from_phaseB_seed44_500m \
    artifacts/baselines \
    runpod_downloads/v0_minimal_active_vision_16x16_race_phaseC_turns_from_phaseB_seed44_500m \
    runpod_downloads/v0_minimal_active_vision_16x16_race_semantic_phaseC_turns_from_phaseB_seed44_500m \
    runs/v0_minimal_active_vision_16x16_race_semantic_phaseB_lapterm_banked_seed44_1b; do
    BASE="$(find "$candidate_root" -type f -name '*.bin' 2>/dev/null | sort -V | tail -n 1 || true)"
    [[ -n "$BASE" ]] && break
  done
fi
export BASE="${BASE:-}"

export BATCH_ID="${BATCH_ID:-v0_minimal_active_vision_16x16_race_swiftlike_from_phaseC_seed44_500m}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"

export TASK=7
export MAX_RINGS="${MAX_RINGS:-7}"
export TARGET_DIST="${TARGET_DIST:-5.0}"
export OOB_RADIUS="${OOB_RADIUS:-18.0}"
export RESET_POS_SCALE="${RESET_POS_SCALE:-0.02}"
export RESET_YAW_RANGE="${RESET_YAW_RANGE:-0.30}"
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

export RACE_TRACK_MODE="${RACE_TRACK_MODE:-2.0}"
export RACE_RESET_START_PROB="${RACE_RESET_START_PROB:-0.45}"
export RACE_RESET_T_MIN="${RACE_RESET_T_MIN:-0.08}"
export RACE_RESET_T_MAX="${RACE_RESET_T_MAX:-0.85}"
export RACE_RESET_LATERAL="${RACE_RESET_LATERAL:-0.70}"
export RACE_RESET_YAW_ERROR_FRAC="${RACE_RESET_YAW_ERROR_FRAC:-0.22}"
export RACE_RESET_SPEED_MIN="${RACE_RESET_SPEED_MIN:-0.5}"
export RACE_RESET_SPEED_MAX="${RACE_RESET_SPEED_MAX:-3.0}"

export ALPHA_DIST="${ALPHA_DIST:-0.45}"
export ALPHA_HOVER="${ALPHA_HOVER:-0.0}"
export ALPHA_SHAPING="${ALPHA_SHAPING:-0.0}"
export ALPHA_OMEGA_XY="${ALPHA_OMEGA_XY:-0.0045}"
export ALPHA_OMEGA_Z="${ALPHA_OMEGA_Z:-0.0045}"
export ALPHA_OMEGA_Z_SQ="${ALPHA_OMEGA_Z_SQ:-0.007}"
export ALPHA_OMEGA_Z_MULT="${ALPHA_OMEGA_Z_MULT:-1.5}"
export ALPHA_ACTION_DELTA="${ALPHA_ACTION_DELTA:-0.012}"
export ALPHA_RESET_ACTION_DELTA="${ALPHA_RESET_ACTION_DELTA:-0.0}"

export ACTION_MODE="${ACTION_MODE:-1}"
export ACTION_SCALE="${ACTION_SCALE:-1.0}"
export NORMALIZED_THRUST_MAX="${NORMALIZED_THRUST_MAX:-0.80}"
export LEARNING_RATE="${LEARNING_RATE:-0.0008}"
export MINIBATCH_SIZE="${MINIBATCH_SIZE:-16384}"
export HORIZON="${HORIZON:-64}"
export POLICY_HIDDEN_SIZE="${POLICY_HIDDEN_SIZE:-128}"
export TOTAL_AGENTS="${TOTAL_AGENTS:-8192}"
export NUM_DRONES="${NUM_DRONES:-8192}"

exec bash experiments/run_minimal_active_vision_ppo.sh
