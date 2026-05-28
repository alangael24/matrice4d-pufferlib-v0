#!/usr/bin/env bash
set -euo pipefail

# 16x16 minimal active vision target-hold at 10m.
# This is still HOVER/task=1, but the target is farther away than the 5m baseline.

export BATCH_ID="${BATCH_ID:-v0_minimal_active_vision_16x16_target10m_seed44_1b}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000000}"

export MINIMAL_VISION_WIDTH=16
export MINIMAL_VISION_HEIGHT=16
export TARGET_DIST="${TARGET_DIST:-10.0}"
export OOB_RADIUS="${OOB_RADIUS:-30.0}"

export MINIMAL_VISION_SIGMA="${MINIMAL_VISION_SIGMA:-0.45}"
export MINIMAL_VISION_DEPTH_GAIN="${MINIMAL_VISION_DEPTH_GAIN:-0.08}"
export MINIMAL_VISION_NOISE="${MINIMAL_VISION_NOISE:-0.02}"
export MINIMAL_VISION_DISTRACTORS="${MINIMAL_VISION_DISTRACTORS:-0.0}"

export TOTAL_AGENTS="${TOTAL_AGENTS:-4096}"
export NUM_DRONES="${NUM_DRONES:-4096}"
export HORIZON="${HORIZON:-32}"
export MINIBATCH_SIZE="${MINIBATCH_SIZE:-8192}"
export LEARNING_RATE="${LEARNING_RATE:-0.003}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
export BUILD_FIRST="${BUILD_FIRST:-1}"

exec bash experiments/run_minimal_active_vision_ppo.sh
