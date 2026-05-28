#!/usr/bin/env bash
set -euo pipefail

export BATCH_ID="${BATCH_ID:-v0_minimal_active_vision_64x64_targetmask_seed44_100m}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
export MINIMAL_VISION_WIDTH=64
export MINIMAL_VISION_HEIGHT=64
export MINIMAL_VISION_SIGMA="${MINIMAL_VISION_SIGMA:-0.06}"
export MINIMAL_VISION_NOISE="${MINIMAL_VISION_NOISE:-0.005}"
export TOTAL_AGENTS="${TOTAL_AGENTS:-1024}"
export NUM_DRONES="${NUM_DRONES:-1024}"
export HORIZON="${HORIZON:-32}"
export MINIBATCH_SIZE="${MINIBATCH_SIZE:-8192}"
export LEARNING_RATE="${LEARNING_RATE:-0.0003}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
export BUILD_FIRST="${BUILD_FIRST:-1}"

exec bash experiments/run_minimal_active_vision_ppo.sh
