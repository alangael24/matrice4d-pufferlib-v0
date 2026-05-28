#!/usr/bin/env bash
set -euo pipefail

# Swift-like 16x16 semantic racing with native predictive visual auxiliary loss.
# Rollout/action inference is unchanged. During CUDA training, a small native
# head predicts t+1 retina moments from hidden state + executed action.

export BATCH_ID="${BATCH_ID:-v0_minimal_active_vision_16x16_race_swiftlike_predaux_seed44_500m}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000000}"

export AUX_VIS_COEF="${AUX_VIS_COEF:-0.04}"
export AUX_VIS_FRAC="${AUX_VIS_FRAC:-0.25}"
export AUX_VIS_OBS_OFFSET="${AUX_VIS_OBS_OFFSET:-32}"
export AUX_VIS_WIDTH="${AUX_VIS_WIDTH:-16}"
export AUX_VIS_HEIGHT="${AUX_VIS_HEIGHT:-16}"
export AUX_VIS_CHANNELS="${AUX_VIS_CHANNELS:-3}"

exec bash experiments/run_minimal_active_vision_16x16_race_swiftlike_ppo.sh
