import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import torch

from pufferlib import _C
from tools.drone_realtime_viewer import (
    DEFAULT_POLICY,
    NativeMinGRUPolicy,
    make_args,
    tensor_from_ptr,
)


DEFAULT_OUTPUT = Path(
    "/mnt/c/Users/alang/matrice4d_experiments/runs/"
    "m4d_v0_2026-04-23_ppo_hover_30M_seed000/eval/"
    "run2_deterministic_eval_cpu.json"
)


def summarize(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main():
    parser = argparse.ArgumentParser(description="Deterministic CPU eval for Matrice 4D native MinGRU checkpoint.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_POLICY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--num-drones", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--hover-target-dist", type=float, default=0.5)
    parser.add_argument("--domain-randomization", type=float, default=0.0)
    parser.add_argument("--action-scale", type=float, default=0.2)
    parser.add_argument("--reset-pos-scale", type=float, default=1.0)
    parser.add_argument("--reset-yaw-range", type=float, default=0.0)
    parser.add_argument("--reset-vel-max", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if getattr(_C, "env_name", None) != "drone":
        raise RuntimeError(f"pufferlib._C is built for {_C.env_name}, not drone. Run: bash build.sh drone --cpu")

    vec = _C.create_vec(
        make_args(
            args.num_drones,
            args.hover_target_dist,
            args.action_scale,
            args.domain_randomization,
            args.reset_pos_scale,
            args.reset_yaw_range,
            args.reset_vel_max,
        ),
        0,
    )
    policy = NativeMinGRUPolicy(checkpoint, hidden_size=args.hidden_size, num_layers=args.num_layers)
    state = policy.initial_state(vec.total_agents)
    obs = tensor_from_ptr(vec.obs_ptr, (vec.total_agents, vec.obs_size))
    rewards = tensor_from_ptr(vec.rewards_ptr, (vec.total_agents,))
    terminals = tensor_from_ptr(vec.terminals_ptr, (vec.total_agents,))

    vec.reset()
    completed = []
    episode_returns = torch.zeros(vec.total_agents)
    episode_lengths = torch.zeros(vec.total_agents)
    total_steps = 0

    try:
        while len(completed) < args.episodes:
            actions, state = policy(obs, state)
            vec.cpu_step(actions.data_ptr())
            total_steps += vec.total_agents

            episode_returns += rewards
            episode_lengths += 1

            done = terminals > 0.0
            if torch.any(done):
                done_indices = torch.nonzero(done, as_tuple=False).flatten().tolist()
                for idx in done_indices:
                    completed.append(
                        {
                            "episode_return": float(episode_returns[idx].item()),
                            "episode_length": float(episode_lengths[idx].item()),
                        }
                    )
                    if len(completed) >= args.episodes:
                        break
                episode_returns[done] = 0.0
                episode_lengths[done] = 0.0
                state[:, done, :] = 0.0

            if total_steps > args.episodes * args.max_steps * vec.total_agents * 4:
                raise RuntimeError("Eval safety stop hit before enough episodes completed")

        logs = vec.log()
    finally:
        vec.close()

    completed = completed[: args.episodes]
    final_metrics = {
        "perf": float(logs.get("perf", 0.0)),
        "score": float(logs.get("score", 0.0)),
        "rings_passed": float(logs.get("rings_passed", 0.0)),
        "ring_collisions": float(logs.get("ring_collisions", 0.0)),
        "collisions": float(logs.get("collisions", 0.0)),
        "oob": float(logs.get("oob", 0.0)),
        "timeout": float(logs.get("timeout", 0.0)),
        "episode_return": float(logs.get("episode_return", 0.0)),
        "episode_length": float(logs.get("episode_length", 0.0)),
        "ema_dist": float(logs.get("ema_dist", 0.0)),
        "ema_vel": float(logs.get("ema_vel", 0.0)),
        "ema_omega": float(logs.get("ema_omega", 0.0)),
    }
    thresholds = {
        "perf_min": 0.95,
        "oob_max": 0.0,
        "timeout_min": 1.0,
        "ema_dist_max": 0.05,
        "ema_vel_max": 0.10,
        "ema_omega_max": 0.20,
    }
    passed = (
        final_metrics["perf"] > thresholds["perf_min"]
        and final_metrics["oob"] <= thresholds["oob_max"]
        and final_metrics["timeout"] >= thresholds["timeout_min"]
        and final_metrics["ema_dist"] < thresholds["ema_dist_max"]
        and final_metrics["ema_vel"] < thresholds["ema_vel_max"]
        and final_metrics["ema_omega"] < thresholds["ema_omega_max"]
    )

    report = {
        "eval_id": "run2_deterministic_eval_cpu",
        "mode": "deterministic_mean_action",
        "checkpoint": str(checkpoint),
        "episodes": len(completed),
        "num_drones": args.num_drones,
        "total_agent_steps": total_steps,
        "config": {
            "hover_target_dist": args.hover_target_dist,
            "domain_randomization": args.domain_randomization,
            "action_scale": args.action_scale,
            "reset_pos_scale": args.reset_pos_scale,
            "reset_yaw_range": args.reset_yaw_range,
            "reset_vel_max": args.reset_vel_max,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
        },
        "final_metrics": final_metrics,
        "episode_summary": {
            "mean_episode_return": summarize([ep["episode_return"] for ep in completed]),
            "mean_episode_length": summarize([ep["episode_length"] for ep in completed]),
        },
        "thresholds": thresholds,
        "passed": passed,
        "notes": "CPU local deterministic eval. Actions use native policy mean output directly, no sampling.",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
