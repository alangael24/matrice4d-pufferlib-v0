#!/usr/bin/env bash
set -euo pipefail

BASE_N2="${BASE:?uso: BASE=/ruta/al/n2_limpio.bin bash experiments/run_n3_physics_batch.sh}"
TOTAL_STAGE="${TOTAL_STAGE:-250000000}"

latest_ckpt() {
  local batch_id="$1"
  find "runs/$batch_id/checkpoints" -type f -name '*.bin' 2>/dev/null | sort -V | tail -n 1
}

set_speed_limiter() {
  local enabled="$1"
  local speed="$2"
  local start="$3"

  ENABLED="$enabled" SPEED="$speed" START="$start" python - <<'PY'
import os
import re
from pathlib import Path

vals = {
    "M4D_TURN_SPEED_LIMIT_ENABLED": int(float(os.environ["ENABLED"])),
    "M4D_TURN_SPEED_LIMIT": float(os.environ["SPEED"]),
    "M4D_TURN_SPEED_START": float(os.environ["START"]),
}

for path in [Path("ocean/drone/drone.h"), Path("ocean/drone/binding_cuda.cu")]:
    s = path.read_text()
    for name, val in vals.items():
        if name.endswith("ENABLED"):
            repl = f"#define {name} {int(val)}"
        else:
            repl = f"#define {name} {val:.6f}f"
        s, n = re.subn(rf"#define {name} [^\n]+", repl, s)
        if n != 1:
            raise SystemExit(f"No pude reemplazar {name} en {path}; matches={n}")
    path.write_text(s)
PY
}

set_gate2_length() {
  local length="$1"
  local original="$2"

  LENGTH="$length" ORIGINAL="$original" python - <<'PY'
import math
import os
import re
from pathlib import Path

G1 = (9.00, 6.45)
G2_ORIG = (8.85, -3.80)
L_ORIG = math.hypot(G2_ORIG[0] - G1[0], G2_ORIG[1] - G1[1])

if os.environ["ORIGINAL"] == "1":
    x, y = G2_ORIG
    L = L_ORIG
else:
    L = float(os.environ["LENGTH"])
    ux = (G2_ORIG[0] - G1[0]) / L_ORIG
    uy = (G2_ORIG[1] - G1[1]) / L_ORIG
    x = G1[0] + L * ux
    y = G1[1] + L * uy

updates = [
    (
        Path("ocean/drone/drone.h"),
        r"case 2: return \(Vec3\)\{[^}]+\};",
        f"case 2: return (Vec3){{ {x:.6f}f, {y:.6f}f, 1.050000f}};",
        r"case 2: return\s+-?[0-9.]+f;[^\n]*",
        "case 2: return -2.26892803f;  // -130 deg",
    ),
    (
        Path("ocean/drone/binding_cuda.cu"),
        r"case 2: return make_float3\([^)]*\);",
        f"case 2: return make_float3( {x:.6f}f, {y:.6f}f, 1.050000f);",
        r"case 2: return\s+-?[0-9.]+f;[^\n]*",
        "case 2: return -2.26892803f; // -130 deg",
    ),
]

for path, pos_pat, pos_line, yaw_pat, yaw_line in updates:
    s = path.read_text()
    s, n = re.subn(pos_pat, pos_line, s, count=1)
    if n != 1:
        raise SystemExit(f"No pude reemplazar gate2 pos en {path}; matches={n}")
    s, n = re.subn(yaw_pat, yaw_line, s, count=1)
    if n != 1:
        raise SystemExit(f"No pude reemplazar gate2 yaw en {path}; matches={n}")
    path.write_text(s)

theta = math.radians(127.10176545)
R = L / (2.0 * math.sin(theta / 2.0))
alat80 = 15.77
vsafe = math.sqrt(alat80 * R)
print(f"G2 length={L:.3f} pos=({x:.3f},{y:.3f}) R={R:.3f} vsafe_cap080={vsafe:.3f}")
PY
}

run_one() {
  local name="$1"
  local base="$2"
  local length="$3"
  local original="$4"
  local speed_enabled="$5"
  local speed_limit="$6"
  local oob_radius="$7"

  echo
  echo "============================================================"
  echo "RUN $name"
  echo "base=$base"
  echo "length=$length original=$original speed_enabled=$speed_enabled speed_limit=$speed_limit oob=$oob_radius"
  echo "============================================================"
  echo

  set_gate2_length "$length" "$original"
  set_speed_limiter "$speed_enabled" "$speed_limit" 12.0

  export BUILD_FIRST=1
  export BASE="$base"
  export BATCH_ID="v0_minimal_active_vision_16x16_race_n3_phys_${name}_seed44_${TOTAL_STAGE}"
  export TOTAL_TIMESTEPS="$TOTAL_STAGE"
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
  export ALPHA_DIST=0.45

  # Keep the required turn nearly unlocked; this batch isolates v/L feasibility.
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

  LAST_CKPT="$(latest_ckpt "$BATCH_ID")"
  if [[ -z "$LAST_CKPT" ]]; then
    echo "ERROR: no encontre checkpoint para $BATCH_ID" >&2
    exit 1
  fi
  echo "LAST_CKPT=$LAST_CKPT"
}

# S1: original with conservative speed cap.
run_one "S1_orig_speedcap8p5_from_n2" \
  "$BASE_N2" 10.251 1 1 8.5 18.0
CKPT_S1="$LAST_CKPT"

# S2: original with nominal physical speed cap.
run_one "S2_orig_speedcap9p5_from_S1" \
  "$CKPT_S1" 10.251 1 1 9.5 18.0
CKPT_S2="$LAST_CKPT"

# L22: no speed cap, but G1->G2 is long enough for v ~= 13.6 m/s.
run_one "L22_nocap_from_n2" \
  "$BASE_N2" 22.0 0 0 99.0 30.0
CKPT_L22="$LAST_CKPT"

# L16: shorten the feasible path.
run_one "L16_nocap_from_L22" \
  "$CKPT_L22" 16.0 0 0 99.0 24.0
CKPT_L16="$LAST_CKPT"

# L12: near-original path length.
run_one "L12_nocap_from_L16" \
  "$CKPT_L16" 12.0 0 0 99.0 20.0
CKPT_L12="$LAST_CKPT"

# L10: original path with nominal speed cap.
run_one "L10_orig_speedcap9p5_from_L12" \
  "$CKPT_L12" 10.251 1 1 9.5 18.0
CKPT_L10="$LAST_CKPT"

# Leave source in original geometry and limiter off after the batch.
set_gate2_length 10.251 1
set_speed_limiter 0 99.0 12.0

echo
echo "Batch fisico completo."
echo "CKPT_S1=$CKPT_S1"
echo "CKPT_S2=$CKPT_S2"
echo "CKPT_L22=$CKPT_L22"
echo "CKPT_L16=$CKPT_L16"
echo "CKPT_L12=$CKPT_L12"
echo "CKPT_L10=$CKPT_L10"
