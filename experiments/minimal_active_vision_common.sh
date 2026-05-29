#!/usr/bin/env bash

mav_default() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name:-}" ]]; then
    export "$name=$value"
  else
    export "$name=${!name}"
  fi
}

mav_race16_base() {
  mav_default TASK 7
  mav_default MAX_RINGS 8
  mav_default TARGET_DIST 5.0
  mav_default OOB_RADIUS 18.0
  mav_default RESET_POS_SCALE 0.08
  mav_default RESET_YAW_RANGE 0.60
  mav_default RESET_VEL_MAX 0.0

  export MINIMAL_VISION_ENABLED=1.0
  mav_default MINIMAL_VISION_ONLY 0.0
  mav_default MINIMAL_VISION_MASK_TARGET 1.0
  export MINIMAL_VISION_SPAWN_VISIBLE_TARGET=1.0
  export MINIMAL_VISION_WIDTH=16
  export MINIMAL_VISION_HEIGHT=16
  mav_default MINIMAL_VISION_FOV 2.0943951
  mav_default MINIMAL_VISION_VFOV 1.3962634
  mav_default MINIMAL_VISION_SIGMA 0.22
  mav_default MINIMAL_VISION_DEPTH_GAIN 0.08
  mav_default MINIMAL_VISION_NOISE 0.02
  mav_default MINIMAL_VISION_DISTRACTORS 0.0

  mav_default ACTION_MODE 1
  mav_default ACTION_SCALE 1.0
  mav_default NORMALIZED_THRUST_MAX 0.80
  mav_default MINIBATCH_SIZE 16384
  mav_default HORIZON 64
  mav_default POLICY_HIDDEN_SIZE 128
  mav_default TOTAL_AGENTS 8192
  mav_default NUM_DRONES 8192
}

mav_race16_reward_defaults() {
  mav_default ALPHA_DIST "$1"
  mav_default ALPHA_HOVER 0.0
  mav_default ALPHA_SHAPING 0.0
  mav_default ALPHA_OMEGA_XY "$2"
  mav_default ALPHA_OMEGA_Z "$3"
  mav_default ALPHA_OMEGA_Z_SQ "$4"
  mav_default ALPHA_OMEGA_Z_MULT 1.5
  mav_default ALPHA_ACTION_DELTA "$5"
  mav_default ALPHA_RESET_ACTION_DELTA 0.0
}

mav_race16_segment_reset_defaults() {
  mav_default RACE_TRACK_MODE 1.0
  export RACE_SEGMENT_MODE=1.0
  mav_default RACE_RESET_T_MIN 0.02
  mav_default RACE_RESET_T_MAX 0.30
  mav_default RACE_RESET_LATERAL 0.45
  mav_default RACE_RESET_YAW_ERROR_FRAC 0.12
  mav_default RACE_RESET_SPEED_MIN 0.4
  mav_default RACE_RESET_SPEED_MAX 2.0
}

mav_exec_base() {
  exec bash experiments/run_minimal_active_vision_ppo.sh
}
