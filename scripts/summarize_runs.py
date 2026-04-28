#!/usr/bin/env python3
"""Summarize PufferLib run stdout/checkpoint artifacts.

This script intentionally parses plain stdout files so it works even if a run
crashes before richer tracking integrations flush metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "perf",
    "score",
    "rings_passed",
    "ring_collisions",
    "collisions",
    "oob",
    "timeout",
    "episode_return",
    "episode_length",
    "ema_dist",
    "ema_vel",
    "ema_omega",
    "ema_omega_x",
    "ema_omega_y",
    "ema_omega_z",
    "mean_abs_action",
    "max_abs_action",
    "action_saturation_frac",
    "motor_clip_low_frac",
    "motor_clip_high_frac",
    "mean_rpm_FL",
    "mean_rpm_FR",
    "mean_rpm_RL",
    "mean_rpm_RR",
    "r_dist",
    "r_hover",
    "r_shaping",
    "r_omega",
    "r_omega_xy",
    "r_omega_z",
    "r_terminal",
    "mass_mult_mean",
    "ixx_mult_mean",
    "iyy_mult_mean",
    "izz_mult_mean",
    "k_thrust_mult_mean",
    "linear_drag_mult_mean",
    "yaw_drag_mult_mean",
    "motor_lag_mult_mean",
    "com_x_mean",
    "com_y_mean",
    "com_z_mean",
]

LOSS_KEYS = ["policy", "value", "entropy", "total", "old_kl", "kl", "clipfrac"]
SUMMARY_KEYS = ["env", "params", "steps", "sps", "epoch"]


def parse_numeric(value: str) -> float | int | str:
    value = value.strip()
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)([KMB])?", value, flags=re.IGNORECASE)
    if not match:
        return value
    base = float(match.group(1))
    suffix = (match.group(2) or "").upper()
    scale = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
    out = base * scale
    return int(out) if out.is_integer() else out


def read_metric(text: str, key: str, word: bool = False) -> Any:
    token = r"[A-Za-z][A-Za-z0-9_-]*" if word else r"-?\d+(?:\.\d+)?(?:[KMB])?"
    matches = re.findall(rf"(?:^|\s){re.escape(key)}\s+({token})(?=\s|$)", text, flags=re.IGNORECASE)
    if not matches:
        return None
    value = matches[-1]
    return value if word else parse_numeric(value)


def parse_command_flags(command: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    if not command:
        return parsed

    mapping = {
        "seed": "seed",
        "train.seed": "seed",
        "train.total-timesteps": "total_timesteps",
        "vec.total-agents": "total_agents",
        "vec.num-buffers": "num_buffers",
        "vec.num-threads": "num_threads",
        "env.num-drones": "num_drones",
        "env.hover-target-dist": "target_dist",
        "env.domain-randomization": "domain_randomization",
        "env.action-scale": "action_scale",
        "env.reset-yaw-range": "reset_yaw_range",
        "env.reset-vel-max": "reset_vel_max",
        "env.reset-pos-scale": "reset_pos_scale",
        "env.oob-radius": "oob_radius",
        "env.alpha-omega-z-mult": "yaw_mult",
        "env.dr-mass": "dr_mass",
        "env.dr-inertia": "dr_inertia",
        "env.dr-k-thrust": "dr_k_thrust",
        "env.dr-linear-drag": "dr_linear_drag",
        "env.dr-yaw-drag": "dr_yaw_drag",
        "env.dr-motor-lag": "dr_motor_lag",
        "env.dr-com-xy": "dr_com_xy",
        "env.dr-com-z": "dr_com_z",
        "env.action-latency": "action_latency",
        "env.sensor-noise": "sensor_noise",
        "policy.num-layers": "num_layers",
    }

    for key, value in re.findall(r"--([A-Za-z0-9_.-]+)\s+([^\\\s]+)", command):
        out_key = mapping.get(key)
        if out_key is not None:
            parsed[out_key] = parse_numeric(value.strip("'\""))
    return parsed


def parse_stdout(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    clean = re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)
    clean = re.sub(r"[ \t]+", " ", clean)

    summary: dict[str, Any] = {}
    for key in SUMMARY_KEYS:
        value = read_metric(clean, key, word=(key == "env"))
        if value is not None:
            summary[key] = value

    losses = {key: value for key in LOSS_KEYS if (value := read_metric(clean, key)) is not None}
    metrics = {key: value for key in METRIC_KEYS if (value := read_metric(clean, key)) is not None}

    return {
        "stdout": str(path),
        "summary": summary,
        "losses": losses,
        "metrics": metrics,
    }


def checkpoint_files(run_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for pattern in ("*.bin", "*.pt", "*.pth", "*.ckpt", "*.safetensors"):
        for file in run_dir.glob(f"checkpoints/**/{pattern}"):
            stat = file.stat()
            files.append({
                "path": str(file),
                "bytes": stat.st_size,
                "mtime": stat.st_mtime,
            })
    return sorted(files, key=lambda item: item["mtime"])


def summarize_run(run_dir: Path) -> dict[str, Any]:
    stdout = run_dir / "stdout.txt"
    parsed = parse_stdout(stdout) if stdout.exists() else {"summary": {}, "losses": {}, "metrics": {}}

    metadata = {}
    metadata_path = run_dir / "run_metadata.env"
    if metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key] = value

    command = ""
    command_flags = {}
    command_path = run_dir / "command.sh"
    if command_path.exists():
        command = command_path.read_text(encoding="utf-8", errors="replace").strip()
        command_flags = parse_command_flags(command)

    checkpoints = checkpoint_files(run_dir)
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "metadata": metadata,
        "command": command,
        "command_flags": command_flags,
        "summary": parsed.get("summary", {}),
        "losses": parsed.get("losses", {}),
        "metrics": parsed.get("metrics", {}),
        "checkpoint_count": len(checkpoints),
        "latest_checkpoint": checkpoints[-1] if checkpoints else None,
        "checkpoints": checkpoints,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "exit_code",
        "seed",
        "total_timesteps",
        "target_dist",
        "action_scale",
        "yaw_mult",
        "domain_randomization",
        "reset_yaw_range",
        "reset_vel_max",
        "reset_pos_scale",
        "oob_radius",
        "total_agents",
        "num_buffers",
        "num_threads",
        "num_drones",
        "dr_mass",
        "dr_inertia",
        "dr_k_thrust",
        "dr_linear_drag",
        "dr_yaw_drag",
        "dr_motor_lag",
        "dr_com_xy",
        "dr_com_z",
        "action_latency",
        "sensor_noise",
        "steps",
        "sps",
        "epoch",
        "perf",
        "score",
        "oob",
        "timeout",
        "episode_return",
        "episode_length",
        "ema_dist",
        "ema_vel",
        "ema_omega",
        "ema_omega_x",
        "ema_omega_y",
        "ema_omega_z",
        "mean_abs_action",
        "max_abs_action",
        "action_saturation_frac",
        "motor_clip_low_frac",
        "motor_clip_high_frac",
        "mean_rpm_FL",
        "mean_rpm_FR",
        "mean_rpm_RL",
        "mean_rpm_RR",
        "r_dist",
        "r_hover",
        "r_shaping",
        "r_omega",
        "r_omega_xy",
        "r_omega_z",
        "r_terminal",
        "mass_mult_mean",
        "ixx_mult_mean",
        "iyy_mult_mean",
        "izz_mult_mean",
        "k_thrust_mult_mean",
        "linear_drag_mult_mean",
        "yaw_drag_mult_mean",
        "motor_lag_mult_mean",
        "com_x_mean",
        "com_y_mean",
        "com_z_mean",
        "checkpoint_count",
        "latest_checkpoint_bytes",
        "latest_checkpoint_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics", {})
            summary = row.get("summary", {})
            metadata = row.get("metadata", {})
            command_flags = row.get("command_flags", {})
            latest = row.get("latest_checkpoint") or {}
            writer.writerow({
                "run_name": row.get("run_name"),
                "exit_code": metadata.get("exit_code"),
                "seed": command_flags.get("seed"),
                "total_timesteps": command_flags.get("total_timesteps"),
                "target_dist": command_flags.get("target_dist"),
                "action_scale": command_flags.get("action_scale"),
                "yaw_mult": command_flags.get("yaw_mult"),
                "domain_randomization": command_flags.get("domain_randomization"),
                "reset_yaw_range": command_flags.get("reset_yaw_range"),
                "reset_vel_max": command_flags.get("reset_vel_max"),
                "reset_pos_scale": command_flags.get("reset_pos_scale"),
                "oob_radius": command_flags.get("oob_radius"),
                "total_agents": command_flags.get("total_agents"),
                "num_buffers": command_flags.get("num_buffers"),
                "num_threads": command_flags.get("num_threads"),
                "num_drones": command_flags.get("num_drones"),
                "dr_mass": command_flags.get("dr_mass"),
                "dr_inertia": command_flags.get("dr_inertia"),
                "dr_k_thrust": command_flags.get("dr_k_thrust"),
                "dr_linear_drag": command_flags.get("dr_linear_drag"),
                "dr_yaw_drag": command_flags.get("dr_yaw_drag"),
                "dr_motor_lag": command_flags.get("dr_motor_lag"),
                "dr_com_xy": command_flags.get("dr_com_xy"),
                "dr_com_z": command_flags.get("dr_com_z"),
                "action_latency": command_flags.get("action_latency"),
                "sensor_noise": command_flags.get("sensor_noise"),
                "steps": summary.get("steps"),
                "sps": summary.get("sps"),
                "epoch": summary.get("epoch"),
                "perf": metrics.get("perf"),
                "score": metrics.get("score"),
                "oob": metrics.get("oob"),
                "timeout": metrics.get("timeout"),
                "episode_return": metrics.get("episode_return"),
                "episode_length": metrics.get("episode_length"),
                "ema_dist": metrics.get("ema_dist"),
                "ema_vel": metrics.get("ema_vel"),
                "ema_omega": metrics.get("ema_omega"),
                "ema_omega_x": metrics.get("ema_omega_x"),
                "ema_omega_y": metrics.get("ema_omega_y"),
                "ema_omega_z": metrics.get("ema_omega_z"),
                "mean_abs_action": metrics.get("mean_abs_action"),
                "max_abs_action": metrics.get("max_abs_action"),
                "action_saturation_frac": metrics.get("action_saturation_frac"),
                "motor_clip_low_frac": metrics.get("motor_clip_low_frac"),
                "motor_clip_high_frac": metrics.get("motor_clip_high_frac"),
                "mean_rpm_FL": metrics.get("mean_rpm_FL"),
                "mean_rpm_FR": metrics.get("mean_rpm_FR"),
                "mean_rpm_RL": metrics.get("mean_rpm_RL"),
                "mean_rpm_RR": metrics.get("mean_rpm_RR"),
                "r_dist": metrics.get("r_dist"),
                "r_hover": metrics.get("r_hover"),
                "r_shaping": metrics.get("r_shaping"),
                "r_omega": metrics.get("r_omega"),
                "r_omega_xy": metrics.get("r_omega_xy"),
                "r_omega_z": metrics.get("r_omega_z"),
                "r_terminal": metrics.get("r_terminal"),
                "mass_mult_mean": metrics.get("mass_mult_mean"),
                "ixx_mult_mean": metrics.get("ixx_mult_mean"),
                "iyy_mult_mean": metrics.get("iyy_mult_mean"),
                "izz_mult_mean": metrics.get("izz_mult_mean"),
                "k_thrust_mult_mean": metrics.get("k_thrust_mult_mean"),
                "linear_drag_mult_mean": metrics.get("linear_drag_mult_mean"),
                "yaw_drag_mult_mean": metrics.get("yaw_drag_mult_mean"),
                "motor_lag_mult_mean": metrics.get("motor_lag_mult_mean"),
                "com_x_mean": metrics.get("com_x_mean"),
                "com_y_mean": metrics.get("com_y_mean"),
                "com_z_mean": metrics.get("com_z_mean"),
                "checkpoint_count": row.get("checkpoint_count"),
                "latest_checkpoint_bytes": latest.get("bytes"),
                "latest_checkpoint_path": latest.get("path"),
            })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    if bool(args.run_dir) == bool(args.runs_root):
        parser.error("Provide exactly one of --run-dir or --runs-root")

    if args.run_dir:
        summary = summarize_run(args.run_dir)
        if args.output:
            write_json(args.output, summary)
        else:
            print(json.dumps(summary, indent=2))
        return 0

    run_dirs = sorted(path for path in args.runs_root.iterdir() if path.is_dir())
    summaries = [summarize_run(path) for path in run_dirs if (path / "stdout.txt").exists()]
    payload = {
        "runs_root": str(args.runs_root),
        "run_count": len(summaries),
        "runs": summaries,
    }
    if args.output:
        write_json(args.output, payload)
    else:
        print(json.dumps(payload, indent=2))
    if args.csv:
        write_csv(args.csv, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
