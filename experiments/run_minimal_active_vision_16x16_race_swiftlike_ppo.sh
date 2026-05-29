#!/usr/bin/env bash
set -euo pipefail

# Phase D: Swift-like 16x16 semantic racing.

source "$(dirname "$0")/minimal_active_vision_common.sh"

if [[ "${FROM_SCRATCH:-0}" == "1" ]]; then
  BASE=""
elif [[ -z "${BASE:-}" ]]; then
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

export MAX_RINGS="${MAX_RINGS:-7}"
export RESET_POS_SCALE="${RESET_POS_SCALE:-0.02}"
export RESET_YAW_RANGE="${RESET_YAW_RANGE:-0.30}"
mav_race16_base

mav_default RACE_TRACK_MODE 2.0
mav_default RACE_RESET_START_PROB 0.45
mav_default RACE_RESET_T_MIN 0.08
mav_default RACE_RESET_T_MAX 0.85
mav_default RACE_RESET_LATERAL 0.70
mav_default RACE_RESET_YAW_ERROR_FRAC 0.22
mav_default RACE_RESET_SPEED_MIN 0.5
mav_default RACE_RESET_SPEED_MAX 3.0

mav_race16_reward_defaults 0.45 0.0045 0.0045 0.007 0.012
mav_default LEARNING_RATE 0.0008

mav_exec_base
