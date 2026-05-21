#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EPISODES="${EPISODES:-8192}"
NUM_AGENTS="${NUM_AGENTS:-64}"
MAX_JOBS="${MAX_JOBS:-3}"
RESET_STATE_INTERVAL="${RESET_STATE_INTERVAL:-32}"
ACTION_SCALE="${ACTION_SCALE:-1.0}"
ENV_SEED="${ENV_SEED:-123}"
RUN_ID="${RUN_ID:-v1_clean_eval_top_seeds_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/runs/$RUN_ID}"
DEFAULT_POLICY_ROOT="$ROOT_DIR/runs/v1_norm_thrust_dr_family_v05_authority_gated_from_B2_5seeds_300m"
if [[ ! -d "$DEFAULT_POLICY_ROOT" && -d "/Users/alan/matrice4d_experiments/runpod_logs/v1_norm_thrust_dr_family_v05_authority_gated_from_B2_5seeds_300m" ]]; then
  DEFAULT_POLICY_ROOT="/Users/alan/matrice4d_experiments/runpod_logs/v1_norm_thrust_dr_family_v05_authority_gated_from_B2_5seeds_300m"
fi
POLICY_ROOT="${POLICY_ROOT:-$DEFAULT_POLICY_ROOT}"

SEEDS="${SEEDS:-46 44 43}"
CONFIGS="${CONFIGS:-nominal narrow20 family_v0.5_authority_gated family_v1_holdout_raw low_authority_holdout motor_tau_high_holdout mass_high_holdout mixed_motors_mild 3plus1_mismatch capped_high_thrust}"

LOG_DIR="$OUT_DIR/logs"
EPISODE_DIR="$OUT_DIR/episodes"
ROW_DIR="$OUT_DIR/rows"
SUMMARY_CSV="$OUT_DIR/summary.csv"
SUMMARY_MD="$OUT_DIR/summary.md"

mkdir -p build "$OUT_DIR" "$LOG_DIR" "$EPISODE_DIR" "$ROW_DIR"

clang -O3 -DNDEBUG -Wall -Werror=return-type -Wno-unused-function -Wno-unused-variable \
  -I./tools -I./src -I./vendor -I./ocean/drone \
  tools/eval_drone_weight.c -lm -lpthread -o build/eval_drone_weight

policy_for_seed() {
  local seed="$1"
  local path
  path="$(find "$POLICY_ROOT" -path "*seed${seed}_300000000/checkpoints*" -name '*.bin' -type f | sort -V | tail -n 1)"
  if [[ -z "$path" || ! -f "$path" ]]; then
    echo "Missing checkpoint for seed ${seed} under ${POLICY_ROOT}" >&2
    return 1
  fi
  printf '%s\n' "$path"
}

throttle_jobs() {
  while true; do
    local running
    running="$(jobs -pr | wc -l | tr -d ' ')"
    if [[ "$running" -lt "$MAX_JOBS" ]]; then
      break
    fi
    sleep 2
  done
}

run_one() {
  local seed="$1"
  local config="$2"
  local policy_path="$3"
  local stem="seed${seed}_${config}"
  local log_path="$LOG_DIR/${stem}.log"
  local episode_csv="$EPISODE_DIR/${stem}_episodes.csv"
  local row_path="$ROW_DIR/${stem}.csv"

  echo "running seed=${seed} config=${config} episodes=${EPISODES} agents=${NUM_AGENTS}" >&2
  env M4D_RESET_STATE_INTERVAL="$RESET_STATE_INTERVAL" \
      M4D_ENV_SEED="$ENV_SEED" \
      M4D_EPISODE_CSV="$episode_csv" \
      ./build/eval_drone_weight "$policy_path" "$EPISODES" "$config" "$ACTION_SCALE" "$NUM_AGENTS" \
      > "$log_path" 2>&1

  local csv_line
  csv_line="$(grep '^csv_clean,' "$log_path" | tail -1)"
  if [[ -z "$csv_line" ]]; then
    echo "Missing csv_clean line for ${stem}; see ${log_path}" >&2
    return 1
  fi

  IFS=',' read -r _ got_config got_scale got_agents got_episodes oob timeout ema_dist ema_vel ema_omega_z mean_abs_action action_saturation_frac mean_abs_delta_action mean_abs_delta_action_p95 reset32_action_jump_mean reset32_action_jump_p95 p95_dist p99_dist p95_omega p99_omega nan_inf_count <<< "$csv_line"

  local gate="holdout"
  if [[ "$got_config" == "nominal" || "$got_config" == "narrow20" ]]; then
    gate="$(awk -v oob="$oob" -v timeout="$timeout" 'BEGIN {print (oob == 0 && timeout == 1) ? "pass" : "fail"}')"
  elif [[ "$got_config" == "family_v0.5_authority_gated" || "$got_config" == "family_v05" ]]; then
    gate="$(awk -v oob="$oob" -v timeout="$timeout" -v sat="$action_saturation_frac" -v delta="$mean_abs_delta_action" 'BEGIN {print (oob <= 0.001 && timeout >= 0.999 && sat <= 0.05 && delta <= 0.10) ? "pass" : "fail"}')"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s","%s","%s"\n' \
    "$seed" "$got_config" "$policy_path" "$got_scale" "$got_agents" "$got_episodes" \
    "$oob" "$timeout" "$ema_dist" "$ema_vel" "$ema_omega_z" "$mean_abs_action" \
    "$action_saturation_frac" "$mean_abs_delta_action" "$mean_abs_delta_action_p95" \
    "$reset32_action_jump_mean" "$reset32_action_jump_p95" "$p95_dist" "$p99_dist" \
    "$p95_omega" "$p99_omega" "$nan_inf_count" "$gate" "$log_path" "$episode_csv" \
    > "$row_path"
  echo "done seed=${seed} config=${config} gate=${gate}" >&2
}

cat > "$OUT_DIR/eval_metadata.env" <<EOF
run_id=$RUN_ID
episodes=$EPISODES
num_agents=$NUM_AGENTS
max_jobs=$MAX_JOBS
reset_state_interval=$RESET_STATE_INTERVAL
action_scale=$ACTION_SCALE
env_seed=$ENV_SEED
seeds=$SEEDS
configs=$CONFIGS
policy_root=$POLICY_ROOT
EOF

for seed in $SEEDS; do
  policy_path="$(policy_for_seed "$seed")"
  for config in $CONFIGS; do
    throttle_jobs
    run_one "$seed" "$config" "$policy_path" &
  done
done

wait

printf 'seed,config,policy_path,action_scale,num_agents,episodes,oob,timeout,ema_dist,ema_vel,ema_omega_z,mean_abs_action,action_saturation_frac,mean_abs_delta_action,mean_abs_delta_action_p95,reset32_action_jump_mean,reset32_action_jump_p95,p95_dist,p99_dist,p95_omega,p99_omega,nan_inf_count,gate,log_path,episode_csv\n' > "$SUMMARY_CSV"
for row in "$ROW_DIR"/*.csv; do
  cat "$row" >> "$SUMMARY_CSV"
done

python3 - "$SUMMARY_CSV" "$SUMMARY_MD" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = list(csv.DictReader(source.open()))
rows.sort(key=lambda r: (int(r["seed"]), r["config"]))

cols = [
    "seed", "config", "gate", "oob", "timeout", "ema_dist", "ema_vel",
    "ema_omega_z", "action_saturation_frac", "mean_abs_delta_action",
    "reset32_action_jump_mean", "reset32_action_jump_p95", "p95_dist",
    "p99_dist", "p95_omega", "p99_omega", "nan_inf_count",
]

lines = [
    "# V1 Clean Eval Top Seeds",
    "",
    f"Source: `{source}`",
    "",
    "| " + " | ".join(cols) + " |",
    "| " + " | ".join(["---"] * len(cols)) + " |",
]
for row in rows:
    lines.append("| " + " | ".join(row.get(col, "") for col in cols) + " |")
lines.append("")
out.write_text("\n".join(lines), encoding="utf-8")
PY

echo "$SUMMARY_CSV"
echo "$SUMMARY_MD"
