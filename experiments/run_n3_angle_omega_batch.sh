#!/usr/bin/env bash
set -euo pipefail

BASE_N2="${BASE:?uso: BASE=/ruta/al/n2_limpio.bin bash experiments/run_n3_angle_omega_batch.sh}"
TOTAL_DIRECT="${TOTAL_DIRECT:-250000000}"
TOTAL_STAGE="${TOTAL_STAGE:-250000000}"

set_gate2_angle() {
  local angle_deg="$1"
  local orig="${2:-0}"

  ANGLE_DEG="$angle_deg" ORIG="$orig" python - <<'PY'
import math
import os
import re
from pathlib import Path

angle_deg = float(os.environ["ANGLE_DEG"])
orig = os.environ.get("ORIG", "0") == "1"

# Original swift-like points, using XY because this curriculum changes the
# horizontal course turn while keeping gate 2 altitude fixed.
G0 = (-0.60, -0.86)
G1 = (9.00, 6.45)
G2 = (8.85, -3.80)

yaw_in = math.atan2(G1[1] - G0[1], G1[0] - G0[0])
L12 = math.hypot(G2[0] - G1[0], G2[1] - G1[1])

orig_out_yaw = math.atan2(G2[1] - G1[1], G2[0] - G1[0])
orig_gate_yaw = math.radians(-130.0)
gate_yaw_offset = orig_gate_yaw - orig_out_yaw

if orig:
    x, y = G2
    gate_yaw = orig_gate_yaw
    label = "original"
else:
    # Desired course turn:
    # outgoing bearing = incoming bearing - requested turn angle.
    out_yaw = yaw_in - math.radians(angle_deg)
    x = G1[0] + L12 * math.cos(out_yaw)
    y = G1[1] + L12 * math.sin(out_yaw)

    # Preserve original mismatch between path bearing and gate normal.
    gate_yaw = out_yaw + gate_yaw_offset
    label = f"{angle_deg:.1f}deg"

updates = [
    (
        Path("ocean/drone/drone.h"),
        r"case 2: return \(Vec3\)\{[^}]+\};",
        f"case 2: return (Vec3){{ {x:.6f}f, {y:.6f}f, 1.050000f}};",
        r"case 2: return\s+-?[0-9.]+f;[^\n]*",
        f"case 2: return {gate_yaw:.8f}f; // gate2 angle curriculum {label}",
    ),
    (
        Path("ocean/drone/binding_cuda.cu"),
        r"case 2: return make_float3\([^)]*\);",
        f"case 2: return make_float3( {x:.6f}f, {y:.6f}f, 1.050000f);",
        r"case 2: return\s+-?[0-9.]+f;[^\n]*",
        f"case 2: return {gate_yaw:.8f}f; // gate2 angle curriculum {label}",
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

turn_rad = math.radians(angle_deg)
omega_req = turn_rad / (L12 / 13.6)
print(
    f"gate2={label} pos=({x:.3f},{y:.3f},1.05) "
    f"gate_yaw={math.degrees(gate_yaw):.2f}deg "
    f"omega_req_at_13p6={omega_req:.3f}rad/s"
)
PY
}

latest_ckpt() {
  local batch_id="$1"
  find "runs/$batch_id/checkpoints" -type f -name '*.bin' 2>/dev/null | sort -V | tail -n 1
}

run_one() {
  local name="$1"
  local base="$2"
  local total="$3"
  local angle="$4"
  local orig="$5"
  local thrust="$6"

  echo
  echo "============================================================"
  echo "RUN $name"
  echo "base=$base"
  echo "angle=$angle orig=$orig thrust=$thrust total=$total"
  echo "============================================================"
  echo

  set_gate2_angle "$angle" "$orig"

  export BUILD_FIRST=1
  export BASE="$base"
  export BATCH_ID="v0_minimal_active_vision_16x16_race_n3_${name}_seed44_${total}"
  export TOTAL_TIMESTEPS="$total"
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

  export NORMALIZED_THRUST_MAX="$thrust"
  export ALPHA_DIST=0.45

  # The original G1->G2 transition requires roughly 2.97 rad/s for ~0.75s
  # at v ~= 13.6 m/s. Keep yaw/action-delta nearly unlocked for this batch.
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

# 0) Same original track, but without punishing the required yaw.
run_one "direct_orig_yawunlock_T080" \
  "$BASE_N2" "$TOTAL_DIRECT" "128.126175972" "1" "0.80"
DIRECT_ORIG_CKPT="$LAST_CKPT"

# 1-5) Angle curriculum. 60 deg is just above observed effective turning.
run_one "turn060_yawunlock_T080_from_n2" \
  "$BASE_N2" "$TOTAL_STAGE" "60" "0" "0.80"
CKPT_60="$LAST_CKPT"

run_one "turn075_yawunlock_T080_from_60" \
  "$CKPT_60" "$TOTAL_STAGE" "75" "0" "0.80"
CKPT_75="$LAST_CKPT"

run_one "turn090_yawunlock_T080_from_75" \
  "$CKPT_75" "$TOTAL_STAGE" "90" "0" "0.80"
CKPT_90="$LAST_CKPT"

run_one "turn105_yawunlock_T080_from_90" \
  "$CKPT_90" "$TOTAL_STAGE" "105" "0" "0.80"
CKPT_105="$LAST_CKPT"

# 6) Original full angle from the angular curriculum.
run_one "turn128_orig_yawunlock_T080_from_105" \
  "$CKPT_105" "$TOTAL_STAGE" "128.126175972" "1" "0.80"
CKPT_128_T080="$LAST_CKPT"

# 7) Same original full angle, lower thrust cap.
run_one "turn128_orig_yawunlock_T065_from_105" \
  "$CKPT_105" "$TOTAL_STAGE" "128.126175972" "1" "0.65"
CKPT_128_T065="$LAST_CKPT"

# Leave source in original geometry after the batch.
set_gate2_angle "128.126175972" "1"

echo
echo "Batch completo."
echo "DIRECT_ORIG_CKPT=$DIRECT_ORIG_CKPT"
echo "CKPT_60=$CKPT_60"
echo "CKPT_75=$CKPT_75"
echo "CKPT_90=$CKPT_90"
echo "CKPT_105=$CKPT_105"
echo "CKPT_128_T080=$CKPT_128_T080"
echo "CKPT_128_T065=$CKPT_128_T065"
