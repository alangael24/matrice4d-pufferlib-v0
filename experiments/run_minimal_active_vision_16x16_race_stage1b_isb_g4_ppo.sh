#!/usr/bin/env bash
set -euo pipefail

# Stage 1b: single-segment racing with Initial State Buffer, oversampling G4.

source "$(dirname "$0")/minimal_active_vision_common.sh"

if [[ -z "${BASE:-}" ]]; then
  for candidate in \
    runs/v0_minimal_active_vision_16x16_race_stage1_segments_from_phaseC_seed44_500m/checkpoints/drone/*/0000000315097088.bin; do
    if [[ -f "$candidate" ]]; then
      BASE="$candidate"
      break
    fi
  done
fi

if [[ -z "${BASE:-}" ]]; then
  for candidate_root in \
    runs/v0_minimal_active_vision_16x16_race_stage1_segments_from_phaseC_seed44_500m \
    runpod_downloads/v0_minimal_active_vision_16x16_race_stage1_segments_from_phaseC_seed44_500m \
    artifacts/baselines \
    runs/v0_minimal_active_vision_16x16_race_phaseC_turns_from_phaseB_seed44_500m \
    runpod_downloads/v0_minimal_active_vision_16x16_race_phaseC_turns_from_phaseB_seed44_500m; do
    BASE="$(find "$candidate_root" -type f -name '*.bin' 2>/dev/null | sort -V | tail -n 1 || true)"
    [[ -n "$BASE" ]] && break
  done
fi
export BASE="${BASE:-}"

export BATCH_ID="${BATCH_ID:-v0_minimal_active_vision_16x16_race_stage1b_isb_g4_from_stage1_seed44_500m}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"

export MAX_RINGS="${MAX_RINGS:-7}"
export OOB_RADIUS="${OOB_RADIUS:-14.0}"
export RESET_POS_SCALE="${RESET_POS_SCALE:-0.02}"
export RESET_YAW_RANGE="${RESET_YAW_RANGE:-0.20}"
mav_race16_base
mav_race16_segment_reset_defaults

mav_default RACE_ISB_ENABLED 1.0
mav_default RACE_ISB_PROB 0.50
mav_default RACE_ISB_MARGIN 0.8
mav_default RACE_ISB_POS_XY 0.45
mav_default RACE_ISB_Z 0.25
mav_default RACE_ISB_ANGLE 0.18
mav_default RACE_ISB_VEL 0.60
mav_default RACE_ISB_OMEGA 0.60
mav_default RACE_HARD_GATE_IDX 4
mav_default RACE_HARD_GATE_PROB 0.40

mav_race16_reward_defaults 0.55 0.0045 0.0045 0.007 0.012
mav_default LEARNING_RATE 0.0008

mav_exec_base
