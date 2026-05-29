#!/usr/bin/env bash
set -euo pipefail

# Stage 1b: single-segment Swift-like racing with gate-mask vision and ISB.

source "$(dirname "$0")/minimal_active_vision_common.sh"

if [[ -z "${BASE:-}" ]]; then
  for candidate in \
    artifacts/baselines/stage1_gatemask_no_privcritic_latest.bin \
    artifacts/baselines/stage1_gatemask_no_privcritic_best.bin; do
    if [[ -f "$candidate" ]]; then
      BASE="$candidate"
      break
    fi
  done
fi

if [[ -z "${BASE:-}" ]]; then
  for candidate_root in \
    runs/v0_minimal_active_vision_16x16_race_stage1_segments_gatemask_no_privcritic_scratch_seed44_500m \
    runpod_downloads/v0_minimal_active_vision_16x16_race_stage1_segments_gatemask_no_privcritic_scratch_seed44_500m; do
    BASE="$(find "$candidate_root" -type f -name '*.bin' 2>/dev/null | sort -V | tail -n 1 || true)"
    [[ -n "$BASE" ]] && break
  done
fi
export BASE="${BASE:-}"

export BATCH_ID="${BATCH_ID:-v0_minimal_active_vision_16x16_race_stage1b_isb_gatemask_no_privcritic_from_stage1_seed44_500m}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"

export MAX_RINGS="${MAX_RINGS:-7}"
export OOB_RADIUS="${OOB_RADIUS:-14.0}"
export RESET_POS_SCALE="${RESET_POS_SCALE:-0.02}"
export RESET_YAW_RANGE="${RESET_YAW_RANGE:-0.20}"
export MINIMAL_VISION_MASK_TARGET=0.0
export MINIMAL_VISION_GATE_MASK=1.0
export MINIMAL_VISION_SIGMA="${MINIMAL_VISION_SIGMA:-0.18}"
export MINIMAL_VISION_NOISE="${MINIMAL_VISION_NOISE:-0.01}"
mav_race16_base
mav_race16_segment_reset_defaults

export RACE_ISB_ENABLED=1.0
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
mav_default LEARNING_RATE 0.0006

mav_exec_base
