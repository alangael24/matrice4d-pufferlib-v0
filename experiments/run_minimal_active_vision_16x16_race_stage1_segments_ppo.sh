#!/usr/bin/env bash
set -euo pipefail

# Stage 1: single-segment racing. Each episode samples one Swift-like segment
# and terminates successfully as soon as the next gate is passed.

source "$(dirname "$0")/minimal_active_vision_common.sh"

if [[ -z "${BASE:-}" ]]; then
  for candidate_root in \
    artifacts/baselines \
    runs/v0_minimal_active_vision_16x16_race_phaseC_turns_from_phaseB_seed44_500m \
    runpod_downloads/v0_minimal_active_vision_16x16_race_phaseC_turns_from_phaseB_seed44_500m \
    runs/v0_minimal_active_vision_16x16_race_swiftlike_debug3_from_phaseC_seed44_500m \
    runpod_downloads/v0_minimal_active_vision_16x16_race_swiftlike_debug3_from_phaseC_seed44_500m; do
    BASE="$(find "$candidate_root" -type f -name '*.bin' 2>/dev/null | sort -V | tail -n 1 || true)"
    [[ -n "$BASE" ]] && break
  done
fi
export BASE="${BASE:-}"

export BATCH_ID="${BATCH_ID:-v0_minimal_active_vision_16x16_race_stage1_segments_from_phaseC_seed44_500m}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"

export MAX_RINGS="${MAX_RINGS:-7}"
export OOB_RADIUS="${OOB_RADIUS:-14.0}"
export RESET_POS_SCALE="${RESET_POS_SCALE:-0.02}"
export RESET_YAW_RANGE="${RESET_YAW_RANGE:-0.20}"
mav_race16_base
mav_race16_segment_reset_defaults
mav_race16_reward_defaults 0.55 0.0045 0.0045 0.007 0.012
mav_default LEARNING_RATE 0.0008

mav_exec_base
