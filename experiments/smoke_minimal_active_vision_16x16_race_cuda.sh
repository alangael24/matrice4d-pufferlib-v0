#!/usr/bin/env bash
set -euo pipefail

# Fast CUDA smoke for task=RACE + 16x16 minimal vision.
# Intended for RunPod before launching the long 1B experiment.

export BATCH_ID="${BATCH_ID:-smoke_minimal_active_vision_16x16_race_cuda}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
export BUILD_FIRST="${BUILD_FIRST:-1}"

export TOTAL_AGENTS="${TOTAL_AGENTS:-1024}"
export NUM_DRONES="${NUM_DRONES:-1024}"
export NUM_BUFFERS="${NUM_BUFFERS:-1}"
export NUM_THREADS="${NUM_THREADS:-16}"
export MINIBATCH_SIZE="${MINIBATCH_SIZE:-2048}"
export HORIZON="${HORIZON:-32}"

export MAX_RINGS="${MAX_RINGS:-3}"
export TARGET_DIST="${TARGET_DIST:-4.0}"
export OOB_RADIUS="${OOB_RADIUS:-12.0}"
export RESET_POS_SCALE="${RESET_POS_SCALE:-0.02}"
export RESET_YAW_RANGE="${RESET_YAW_RANGE:-0.20}"
export MINIMAL_VISION_SIGMA="${MINIMAL_VISION_SIGMA:-0.25}"

exec bash experiments/run_minimal_active_vision_16x16_race_ppo.sh
