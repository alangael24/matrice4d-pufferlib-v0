#!/usr/bin/env bash
set -euo pipefail

BASE_CKPT="${BASE:?uso: BASE=/ruta/al/n2_limpio.bin bash experiments/run_n3_math_batch.sh}"
TOTAL_PER_EXP="${TOTAL_PER_EXP:-300000000}"

set_math_constants() {
  local vsafe="$1"
  local turn_start="$2"
  local speed_k="$3"
  local center_k="$4"
  local corridor="$5"
  local lookahead_frac="$6"

  M4D_VSAFE="$vsafe" \
  M4D_TURN_START="$turn_start" \
  M4D_SPEED_K="$speed_k" \
  M4D_CENTER_K="$center_k" \
  M4D_CORRIDOR="$corridor" \
  M4D_LOOKAHEAD="$lookahead_frac" \
  python - <<'PY'
import os
import re
from pathlib import Path

paths = [Path("ocean/drone/drone.h"), Path("ocean/drone/binding_cuda.cu")]
vals = {
    "M4D_RACE_TURN_VSAFE": float(os.environ["M4D_VSAFE"]),
    "M4D_RACE_TURN_START": float(os.environ["M4D_TURN_START"]),
    "M4D_RACE_SPEED_K": float(os.environ["M4D_SPEED_K"]),
    "M4D_RACE_CENTER_K": float(os.environ["M4D_CENTER_K"]),
    "M4D_RACE_CENTER_CORRIDOR": float(os.environ["M4D_CORRIDOR"]),
    "M4D_RACE_LOOKAHEAD_FRAC": float(os.environ["M4D_LOOKAHEAD"]),
}

def c_float(val):
    text = f"{val:.7g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text + "f"

for path in paths:
    s = path.read_text()
    for name, val in vals.items():
        new = f"#define {name} {c_float(val)}"
        s, n = re.subn(rf"#define {name} [^\n]+", new, s)
        if n != 1:
            raise SystemExit(f"No pude reemplazar {name} en {path}; matches={n}")
    path.write_text(s)
PY
}

run_one() {
  local name="$1"
  local vsafe="$2"
  local turn_start="$3"
  local speed_k="$4"
  local center_k="$5"
  local corridor="$6"
  local lookahead="$7"
  local alpha_dist="$8"
  local oob_radius="$9"

  echo
  echo "============================================================"
  echo "RUN $name"
  echo "vsafe=$vsafe turn_start=$turn_start speed_k=$speed_k center_k=$center_k corridor=$corridor lookahead=$lookahead alpha_dist=$alpha_dist oob=$oob_radius"
  echo "============================================================"
  echo

  set_math_constants "$vsafe" "$turn_start" "$speed_k" "$center_k" "$corridor" "$lookahead"

  export BUILD_FIRST=1
  export BASE="$BASE_CKPT"
  export BATCH_ID="v0_minimal_active_vision_16x16_race_n3_math_${name}_from_n2_seed44_${TOTAL_PER_EXP}"
  export TOTAL_TIMESTEPS="$TOTAL_PER_EXP"
  export CHECKPOINT_INTERVAL=5

  export MAX_RINGS=3
  export RACE_SEGMENT_MODE=0.0
  export RACE_TRACK_MODE=1.0
  export OOB_RADIUS="$oob_radius"

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
  export ALPHA_DIST="$alpha_dist"
  export ALPHA_HOVER=0.0
  export ALPHA_SHAPING=0.0

  export RACE_ISB_ENABLED=0.0
  export RACE_HARD_GATE_PROB=0.0
  export PRIVILEGED_CRITIC=0.0

  export HORIZON=256
  export MINIBATCH_SIZE=32768
  export LEARNING_RATE=0.00025
  export GPU_ID="${GPU_ID:-0}"

  bash experiments/run_minimal_active_vision_16x16_race_swiftlike_ppo.sh
}

run_one "A_v8p0_start9_center002_speed005_oob18" \
  8.0 9.0 0.050 0.020 1.50 0.45 0.25 18.0

run_one "B_v9p5_start9_center0015_speed0035_oob18" \
  9.5 9.0 0.035 0.015 1.50 0.45 0.30 18.0

run_one "C_v7p2_start12_center0025_speed0065_oob18" \
  7.2 12.0 0.065 0.025 1.50 0.55 0.20 18.0

run_one "D_v8p0_start9_center002_speed005_oob24" \
  8.0 9.0 0.050 0.020 1.50 0.45 0.25 24.0
