#!/usr/bin/env bash
set -euo pipefail

# Minimal PPO-PAL ablation matrix. Defaults are intentionally shorter than the
# main run so we can test whether probes/smooth anchors help before spending a
# full 500M-1B run.

BASE="${BASE:-}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-44}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"

common=(
  BASE="$BASE"
  GPU_ID="$GPU_ID"
  SEED="$SEED"
  TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS"
  BUILD_FIRST="${BUILD_FIRST:-0}"
)

echo "### A: v0.14 + MinGRU/selective DR only"
env "${common[@]}" \
  BATCH_ID="v0_15_ppo_pal_A_gru_selective_from_v014_seed${SEED}_${TOTAL_TIMESTEPS}" \
  PAL_PROBE_PROB=0.0 \
  PAL_PROBE_STEPS=0 \
  PAL_PROBE_AMP=0.0 \
  ALPHA_ACTION_DELTA=0.0 \
  ALPHA_RESET_ACTION_DELTA=0.0 \
  bash experiments/run_v0_15_ppo_pal_from_v014.sh

echo "### B: + smooth/reset penalty"
env "${common[@]}" \
  BATCH_ID="v0_15_ppo_pal_B_smooth_from_v014_seed${SEED}_${TOTAL_TIMESTEPS}" \
  PAL_PROBE_PROB=0.0 \
  PAL_PROBE_STEPS=0 \
  PAL_PROBE_AMP=0.0 \
  ALPHA_ACTION_DELTA=0.005 \
  ALPHA_RESET_ACTION_DELTA=0.010 \
  bash experiments/run_v0_15_ppo_pal_from_v014.sh

echo "### C: + probe-conditioned rollout"
env "${common[@]}" \
  BATCH_ID="v0_15_ppo_pal_C_probe_from_v014_seed${SEED}_${TOTAL_TIMESTEPS}" \
  PAL_PROBE_PROB=0.50 \
  PAL_PROBE_STEPS=16 \
  PAL_PROBE_AMP=0.030 \
  ALPHA_ACTION_DELTA=0.0 \
  ALPHA_RESET_ACTION_DELTA=0.0 \
  bash experiments/run_v0_15_ppo_pal_from_v014.sh

echo "### D: full PPO-PAL pilot"
env "${common[@]}" \
  BATCH_ID="v0_15_ppo_pal_D_probe_smooth_anchor_from_v014_seed${SEED}_${TOTAL_TIMESTEPS}" \
  PAL_PROBE_PROB=0.50 \
  PAL_PROBE_STEPS=16 \
  PAL_PROBE_AMP=0.030 \
  ALPHA_ACTION_DELTA=0.005 \
  ALPHA_RESET_ACTION_DELTA=0.010 \
  bash experiments/run_v0_15_ppo_pal_from_v014.sh
