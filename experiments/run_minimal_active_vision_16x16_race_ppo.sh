#!/usr/bin/env bash
set -euo pipefail

# 16x16 minimal-vision racing experiment.
#
# Uses task=RACE (7), masked target state, and a CUDA race course whose first
# gate starts in the camera FOV. Later gates use bounded yaw/pitch/height deltas
# so the next gate remains visible or recoverable under the nominal path.
# Phase B reset curriculum: 90% start/gate1 and 10% easy visible mid-segment.
# Hard post-gate and out-of-FOV reacquisition resets are disabled for now.
# In RACE, the minimal vision RGB channels are semantic lookahead:
# R=current gate, G=next gate, B=next-next gate.
# Conservative reward/control defaults reduce the high-speed collision mode:
# lower progress/pass incentive, stronger spin/slew penalties, lower thrust cap.
# Race rewards are banked in CPU/CUDA:
# pass_immediate=+0.2, gate_bank+=1.0, lap/timeout_safe pays bank,
# ring_collision=-2, oob=-10 - gate_bank - 0.05*speed^2.
# Pass BASE=/path/to/16x16_hover_checkpoint.bin to continue from a hover policy.

export BATCH_ID="${BATCH_ID:-v0_minimal_active_vision_16x16_race_seed44_1b}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"

export TASK=7
export MAX_RINGS="${MAX_RINGS:-8}"
export TARGET_DIST="${TARGET_DIST:-4.0}"
export OOB_RADIUS="${OOB_RADIUS:-18.0}"
export RESET_POS_SCALE="${RESET_POS_SCALE:-0.08}"
export RESET_YAW_RANGE="${RESET_YAW_RANGE:-0.60}"
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

export ALPHA_DIST="${ALPHA_DIST:-0.5}"
export ALPHA_HOVER="${ALPHA_HOVER:-0.0}"
export ALPHA_SHAPING="${ALPHA_SHAPING:-0.0}"
export ALPHA_OMEGA_XY="${ALPHA_OMEGA_XY:-0.004}"
export ALPHA_OMEGA_Z="${ALPHA_OMEGA_Z:-0.004}"
export ALPHA_OMEGA_Z_SQ="${ALPHA_OMEGA_Z_SQ:-0.006}"
export ALPHA_OMEGA_Z_MULT="${ALPHA_OMEGA_Z_MULT:-1.5}"
export ALPHA_ACTION_DELTA="${ALPHA_ACTION_DELTA:-0.01}"
export ALPHA_RESET_ACTION_DELTA="${ALPHA_RESET_ACTION_DELTA:-0.0}"

export ACTION_MODE="${ACTION_MODE:-1}"
export ACTION_SCALE="${ACTION_SCALE:-1.0}"
export NORMALIZED_THRUST_MAX="${NORMALIZED_THRUST_MAX:-0.80}"
export LEARNING_RATE="${LEARNING_RATE:-0.003}"
export MINIBATCH_SIZE="${MINIBATCH_SIZE:-16384}"
export HORIZON="${HORIZON:-64}"
export POLICY_HIDDEN_SIZE="${POLICY_HIDDEN_SIZE:-128}"
export TOTAL_AGENTS="${TOTAL_AGENTS:-8192}"
export NUM_DRONES="${NUM_DRONES:-8192}"

exec bash experiments/run_minimal_active_vision_ppo.sh
