#!/usr/bin/env bash
set -euo pipefail

BASE_N2="${BASE:?uso: BASE=/ruta/al/n2_limpio.bin bash experiments/run_n3_observability_batch.sh}"
TOTAL_DIAG="${TOTAL_DIAG:-250000000}"
TOTAL_STAGE="${TOTAL_STAGE:-300000000}"

latest_ckpt() {
  local batch_id="$1"
  find "runs/$batch_id/checkpoints" -type f -name '*.bin' 2>/dev/null | sort -V | tail -n 1
}

set_nav_tutor() {
  local enabled="$1"
  local gain="$2"
  local dist_scale="${3:-30.0}"

  ENABLED="$enabled" GAIN="$gain" DIST_SCALE="$dist_scale" python - <<'PY'
import os
import re
from pathlib import Path

vals = {
    "M4D_NAV_TUTOR_ENABLED": int(float(os.environ["ENABLED"])),
    "M4D_NAV_TUTOR_GAIN": float(os.environ["GAIN"]),
    "M4D_NAV_TUTOR_DIST_SCALE": float(os.environ["DIST_SCALE"]),
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

run_common() {
  local name="$1"
  local base="$2"
  local total="$3"
  local mask_target="$4"
  local tutor_enabled="$5"
  local tutor_gain="$6"

  echo
  echo "============================================================"
  echo "RUN $name"
  echo "base=$base"
  echo "mask_target=$mask_target tutor_enabled=$tutor_enabled tutor_gain=$tutor_gain"
  echo "============================================================"
  echo

  set_nav_tutor "$tutor_enabled" "$tutor_gain" 30.0

  export BUILD_FIRST=1
  export BASE="$base"
  export BATCH_ID="v0_minimal_active_vision_16x16_race_n3_obs_${name}_seed44_${total}"
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
  export MINIMAL_VISION_MASK_TARGET="$mask_target"
  export MINIMAL_VISION_GATE_MASK=1.0

  export NORMALIZED_THRUST_MAX=0.80
  export ALPHA_DIST=0.45

  # Do not punish the maneuver while testing observability.
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

# T0: no custom tutor; only unmask the original obs[10..18] target fields.
run_common "T0_unmask_current_target_from_n2" \
  "$BASE_N2" "$TOTAL_DIAG" 0.0 0 0.0
CKPT_T0="$LAST_CKPT"

# T1: current+next gate body-frame tutor.
run_common "T1_nav_tutor_gain1_from_n2" \
  "$BASE_N2" "$TOTAL_STAGE" 1.0 1 1.0
CKPT_T1="$LAST_CKPT"

# T2/T3/T4: distillation schedule back toward vision-only.
run_common "T2_nav_tutor_gain0p5_from_T1" \
  "$CKPT_T1" "$TOTAL_STAGE" 1.0 1 0.5
CKPT_T2="$LAST_CKPT"

run_common "T3_nav_tutor_gain0p25_from_T2" \
  "$CKPT_T2" "$TOTAL_STAGE" 1.0 1 0.25
CKPT_T3="$LAST_CKPT"

run_common "T4_vision_only_gain0_from_T3" \
  "$CKPT_T3" "$TOTAL_STAGE" 1.0 1 0.0
CKPT_T4="$LAST_CKPT"

# Restore tutor off at the end.
set_nav_tutor 0 0.0 30.0

echo
echo "Batch observabilidad completo."
echo "CKPT_T0=$CKPT_T0"
echo "CKPT_T1=$CKPT_T1"
echo "CKPT_T2=$CKPT_T2"
echo "CKPT_T3=$CKPT_T3"
echo "CKPT_T4=$CKPT_T4"
