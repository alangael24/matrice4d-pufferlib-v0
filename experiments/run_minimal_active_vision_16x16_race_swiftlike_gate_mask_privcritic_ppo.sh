#!/usr/bin/env bash
set -euo pipefail

# Phase D variant: Swift-like 16x16 gate-edge mask with an asymmetric
# actor/critic contract. The actor gets proprioception plus the 16x16 visual
# gate mask, but the perfect target vector/normal are masked from actor input.
# The critic receives the full first DRONE_STATE_OBS_SIZE state block.

export BATCH_ID="${BATCH_ID:-v0_minimal_active_vision_16x16_race_swiftlike_gatemask_privcritic_seed44_500m}"

export MINIMAL_VISION_ONLY=0.0
export MINIMAL_VISION_MASK_TARGET=0.0
export MINIMAL_VISION_GATE_MASK=1.0
export MINIMAL_VISION_SIGMA="${MINIMAL_VISION_SIGMA:-0.18}"
export MINIMAL_VISION_NOISE="${MINIMAL_VISION_NOISE:-0.01}"

export PRIVILEGED_CRITIC=1.0
export PRIVILEGED_CRITIC_OBS_DIM=23
export PRIVILEGED_CRITIC_HIDDEN="${PRIVILEGED_CRITIC_HIDDEN:-128}"
export ACTOR_OBS_MASK_PREFIX=0
export ACTOR_OBS_MASK_TARGET=1.0

exec bash experiments/run_minimal_active_vision_16x16_race_swiftlike_ppo.sh
