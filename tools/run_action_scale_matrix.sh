#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EPISODES="${EPISODES:-8192}"
NUM_AGENTS="${NUM_AGENTS:-64}"
RESET_STATE_INTERVAL="${RESET_STATE_INTERVAL:-32}"
OUT_DIR="${OUT_DIR:-/Users/alan/matrice4d_experiments/reports}"
RUN_ID="${RUN_ID:-matrix_a_action_scale_sweep_8192}"
OUT_CSV="${OUT_CSV:-$OUT_DIR/${RUN_ID}.csv}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/${RUN_ID}_logs}"

POLICY_A="${POLICY_A:-/Users/alan/matrice4d_experiments/release_assets/v0.14-dr-medium-robust-baseline/v0_14_dr_medium_robust_seed44_scale0.7_reset32_latest.bin}"
POLICY_B="${POLICY_B:-/Users/alan/matrice4d_experiments/release_assets/v0.10-target5-baseline/v0_10_target5_seed46_scale0.5_yawMult5_vec4096x8x32_100m_latest.bin}"

mkdir -p "$OUT_DIR" "$LOG_DIR" build

clang -O3 -DNDEBUG -Wall -Werror=return-type -Wno-unused-function -Wno-unused-variable \
  -I./tools -I./src -I./vendor -I./ocean/drone \
  tools/eval_drone_weight.c -lm -lpthread -o build/eval_drone_weight

printf 'policy,policy_path,config,action_scale,num_agents,episodes,mean_return,std_return,min_return,max_return,timeout_rate,oob_rate,mean_len,deterministic,reset_state_interval\n' > "$OUT_CSV"

run_one() {
  local policy_name="$1"
  local policy_path="$2"
  local config="$3"
  local action_scale="$4"
  local log_path="$LOG_DIR/${policy_name}_${config}_scale${action_scale}.log"

  echo "running policy=${policy_name} config=${config} action_scale=${action_scale} episodes=${EPISODES} num_agents=${NUM_AGENTS} reset_state_interval=${RESET_STATE_INTERVAL}" >&2
  local csv_line
  csv_line="$(
    env M4D_RESET_STATE_INTERVAL="$RESET_STATE_INTERVAL" \
      ./build/eval_drone_weight "$policy_path" "$EPISODES" "$config" "$action_scale" "$NUM_AGENTS" \
      | tee "$log_path" \
      | grep '^csv,' \
      | tail -1
  )"

  IFS=',' read -r _ got_config got_scale got_agents got_episodes mean_return std_return min_return max_return timeout_rate oob_rate mean_len <<< "$csv_line"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1,%s\n' \
    "$policy_name" "$policy_path" "$got_config" "$got_scale" "$got_agents" "$got_episodes" \
    "$mean_return" "$std_return" "$min_return" "$max_return" "$timeout_rate" "$oob_rate" "$mean_len" \
    "$RESET_STATE_INTERVAL" \
    >> "$OUT_CSV"
}

for policy_name in policy_a policy_b; do
  if [[ "$policy_name" == "policy_a" ]]; then
    policy_path="$POLICY_A"
  else
    policy_path="$POLICY_B"
  fi

  for config in baseline light medium; do
    for action_scale in 0.5 0.7 0.8; do
      run_one "$policy_name" "$policy_path" "$config" "$action_scale"
    done
  done
done

echo "$OUT_CSV"
