#!/usr/bin/env bash
set -euo pipefail

# Stage 1b: single-segment racing with an Initial State Buffer (ISB).
#
# Successful, clean gate passes populate a 10-state buffer for the next target
# gate. Resets sample from that buffer with perturbations, then fall back to the
# geometric segment reset while the buffer is empty. Gate 4 is oversampled
# because Stage 1 concentrated most failures there.

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

export TASK=7
export MAX_RINGS="${MAX_RINGS:-7}"
export TARGET_DIST="${TARGET_DIST:-5.0}"
export OOB_RADIUS="${OOB_RADIUS:-14.0}"
export RESET_POS_SCALE="${RESET_POS_SCALE:-0.02}"
export RESET_YAW_RANGE="${RESET_YAW_RANGE:-0.20}"
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

export RACE_TRACK_MODE="${RACE_TRACK_MODE:-1.0}"
export RACE_SEGMENT_MODE=1.0
export RACE_RESET_T_MIN="${RACE_RESET_T_MIN:-0.02}"
export RACE_RESET_T_MAX="${RACE_RESET_T_MAX:-0.30}"
export RACE_RESET_LATERAL="${RACE_RESET_LATERAL:-0.45}"
export RACE_RESET_YAW_ERROR_FRAC="${RACE_RESET_YAW_ERROR_FRAC:-0.12}"
export RACE_RESET_SPEED_MIN="${RACE_RESET_SPEED_MIN:-0.4}"
export RACE_RESET_SPEED_MAX="${RACE_RESET_SPEED_MAX:-2.0}"

export RACE_ISB_ENABLED="${RACE_ISB_ENABLED:-1.0}"
export RACE_ISB_PROB="${RACE_ISB_PROB:-0.50}"
export RACE_ISB_MARGIN="${RACE_ISB_MARGIN:-0.8}"
export RACE_ISB_POS_XY="${RACE_ISB_POS_XY:-0.45}"
export RACE_ISB_Z="${RACE_ISB_Z:-0.25}"
export RACE_ISB_ANGLE="${RACE_ISB_ANGLE:-0.18}"
export RACE_ISB_VEL="${RACE_ISB_VEL:-0.60}"
export RACE_ISB_OMEGA="${RACE_ISB_OMEGA:-0.60}"
export RACE_HARD_GATE_IDX="${RACE_HARD_GATE_IDX:-4}"
export RACE_HARD_GATE_PROB="${RACE_HARD_GATE_PROB:-0.40}"

export ALPHA_DIST="${ALPHA_DIST:-0.55}"
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
