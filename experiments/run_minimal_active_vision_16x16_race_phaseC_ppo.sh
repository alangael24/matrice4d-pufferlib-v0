#!/usr/bin/env bash
set -euo pipefail

# Phase C: harder 16x16 semantic racing.
#
# Continues from the Phase B/lap-terminal baseline when present, then increases
# yaw, pitch/height variation, gate spacing, and reset difficulty.

source "$(dirname "$0")/minimal_active_vision_common.sh"

if [[ -z "${BASE:-}" ]]; then
  BASE="$(find runs/v0_minimal_active_vision_16x16_race_semantic_phaseB_lapterm_banked_seed44_1b \
    -type f -name '*.bin' 2>/dev/null | sort -V | tail -n 1 || true)"
fi
export BASE="${BASE:-}"

export BATCH_ID="${BATCH_ID:-v0_minimal_active_vision_16x16_race_semantic_phaseC_turns_from_phaseB_seed44_500m}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"

mav_race16_base
mav_default MAX_RINGS 8

mav_default RACE_COURSE_YAW_DELTA 0.70
mav_default RACE_COURSE_PITCH_DELTA 0.32
mav_default RACE_COURSE_PITCH_LIMIT 0.45
mav_default RACE_COURSE_SPACING_MIN 4.0
mav_default RACE_COURSE_SPACING_MAX 6.0
mav_default RACE_COURSE_DZ_MAX 0.70

mav_default RACE_RESET_START_PROB 0.60
mav_default RACE_RESET_T_MIN 0.10
mav_default RACE_RESET_T_MAX 0.70
mav_default RACE_RESET_LATERAL 0.55
mav_default RACE_RESET_YAW_ERROR_FRAC 0.18
mav_default RACE_RESET_SPEED_MIN 0.5
mav_default RACE_RESET_SPEED_MAX 2.2

mav_race16_reward_defaults 0.5 0.004 0.004 0.006 0.01
mav_default LEARNING_RATE 0.001

mav_exec_base
