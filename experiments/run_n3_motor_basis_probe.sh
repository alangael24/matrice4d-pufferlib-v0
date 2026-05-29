#!/usr/bin/env bash
set -euo pipefail

BASE_N2="${BASE:?uso: BASE=/ruta/al/n2_limpio.bin bash experiments/run_n3_motor_basis_probe.sh}"
TOTAL_STAGE="${TOTAL_STAGE:-80000000}"

set_assist() {
  local enabled="$1"
  local pattern="$2"
  local gain="$3"
  local min_idx="$4"
  local lookahead="$5"

  ENABLED="$enabled" PATTERN="$pattern" GAIN="$gain" MIN_IDX="$min_idx" LOOKAHEAD="$lookahead" python - <<'PY'
import os
import re
from pathlib import Path

vals_int = {
    "M4D_TURN_ASSIST_ENABLED": int(float(os.environ["ENABLED"])),
    "M4D_TURN_ASSIST_PATTERN": int(float(os.environ["PATTERN"])),
    "M4D_TURN_ASSIST_MIN_IDX": int(float(os.environ["MIN_IDX"])),
    "M4D_TURN_ASSIST_LOOKAHEAD": int(float(os.environ["LOOKAHEAD"])),
}
vals_float = {
    "M4D_TURN_ASSIST_GAIN": float(os.environ["GAIN"]),
}

for path in [Path("ocean/drone/drone.h"), Path("ocean/drone/binding_cuda.cu")]:
    s = path.read_text()
    for name, val in vals_int.items():
        s, n = re.subn(rf"#define {name} [^\n]+", f"#define {name} {val}", s)
        if n != 1:
            raise SystemExit(f"No pude reemplazar {name} en {path}; matches={n}")

    for name, val in vals_float.items():
        s, n = re.subn(rf"#define {name} [^\n]+", f"#define {name} {val:.6f}f", s)
        if n != 1:
            raise SystemExit(f"No pude reemplazar {name} en {path}; matches={n}")

    path.write_text(s)
PY
}

run_one() {
  local name="$1"
  local pattern="$2"
  local min_idx="$3"
  local lookahead="$4"

  echo
  echo "============================================================"
  echo "RUN $name pattern=$pattern min_idx=$min_idx lookahead=$lookahead"
  echo "============================================================"
  echo

  set_assist 1 "$pattern" 0.20 "$min_idx" "$lookahead"

  export BUILD_FIRST=1
  export BASE="$BASE_N2"
  export BATCH_ID="v0_minimal_active_vision_16x16_race_n3_basis_${name}_seed44_${TOTAL_STAGE}"
  export TOTAL_TIMESTEPS="$TOTAL_STAGE"
  export CHECKPOINT_INTERVAL=5

  export MAX_RINGS=3
  export RACE_SEGMENT_MODE=0.0
  export RACE_TRACK_MODE=1.0
  export OOB_RADIUS=18.0

  export RACE_RESET_START_PROB=1.0
  export RACE_RESET_T_MIN=0.00
  export RACE_RESET_T_MAX=0.05
  export RACE_RESET_LATERAL=0.20
  export RACE_RESET_YAW_ERROR_FRAC=0.05
  export RACE_RESET_SPEED_MIN=0.4
  export RACE_RESET_SPEED_MAX=1.5

  export MINIMAL_VISION_FOV=2.0943951
  export MINIMAL_VISION_VFOV=1.3962634
  export MINIMAL_VISION_MASK_TARGET=1.0
  export MINIMAL_VISION_GATE_MASK=1.0

  export NORMALIZED_THRUST_MAX=0.80
  export ALPHA_DIST=0.45

  # Do not punish the diagnostic maneuver while probing motor basis.
  export ALPHA_OMEGA_XY=0.0005
  export ALPHA_OMEGA_Z=0.0
  export ALPHA_OMEGA_Z_SQ=0.0
  export ALPHA_OMEGA_Z_MULT=0.0
  export ALPHA_ACTION_DELTA=0.001
  export ALPHA_RESET_ACTION_DELTA=0.0

  export RACE_ISB_ENABLED=0.0
  export RACE_HARD_GATE_PROB=0.0
  export PRIVILEGED_CRITIC=0.0

  export HORIZON=256
  export MINIBATCH_SIZE=32768
  export LEARNING_RATE=0.00025

  bash experiments/run_minimal_active_vision_16x16_race_swiftlike_ppo.sh
}

# Post-switch: only when current target is G2.
run_one "post_roll_pos"  0 2 0
run_one "post_roll_neg"  1 2 0
run_one "post_pitch_pos" 2 2 0
run_one "post_pitch_neg" 3 2 0
run_one "post_yaw_pos"   4 2 0
run_one "post_yaw_neg"   5 2 0

set_assist 0 0 0.0 2 0

echo "Motor-basis probe completo."
