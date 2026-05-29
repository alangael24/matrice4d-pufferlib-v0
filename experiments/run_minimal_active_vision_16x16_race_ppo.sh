#!/usr/bin/env bash
set -euo pipefail

# 16x16 minimal-vision racing experiment.
#
# Uses task=RACE (7), masked target state, 16x16 RGB retina, semantic gate
# lookahead, banked race rewards, and conservative low-level action defaults.

source "$(dirname "$0")/minimal_active_vision_common.sh"

export BATCH_ID="${BATCH_ID:-v0_minimal_active_vision_16x16_race_seed44_1b}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"

export TARGET_DIST="${TARGET_DIST:-4.0}"
mav_race16_base
mav_race16_reward_defaults 0.5 0.004 0.004 0.006 0.01
mav_default LEARNING_RATE 0.003

mav_exec_base
