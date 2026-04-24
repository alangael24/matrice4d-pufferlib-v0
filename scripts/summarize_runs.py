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
    "mean_abs_action_raw",
    "max_abs_action_raw",
    "raw_action_clip_frac",
    "mean_abs_action_clipped",
    "max_abs_action_clipped",
    "clipped_action_saturation_frac",
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
    "r_terminal",
]

LOSS_KEYS = ["policy", "value", "entropy", "total", "old_kl", "kl", "clipfrac"]
SUMMARY_KEYS = ["env", "params", "steps", "sps", "epoch"]
CSV_FIELDNAMES = [
    "run_name",
    "policy",
    "value",
    "entropy",
    "total",
    "old_kl",
    "kl",
    "clipfrac",
    "steps",
    "epoch",
    "timesteps",
    "seed",
    "target_dist",
    "action_scale",
    "domain_randomization",
    "reset_yaw_range",
    "reset_vel_max",
    "reset_pos_scale",
    "exit_code",
    "checkpoint_count",
    "latest_checkpoint_path",
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
    "mean_abs_action_raw",
    "max_abs_action_raw",
    "raw_action_clip_frac",
    "mean_abs_action_clipped",
    "max_abs_action_clipped",
    "clipped_action_saturation_frac",
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
    "r_terminal",
]


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
    matches = list(re.finditer(rf"(?:^|\s){re.escape(key)}\s+({token})(?=\s|$)", text, flags=re.IGNORECASE))
    if not matches:
        return None
    match = matches[-1]
    return match.group(1) if word else parse_numeric(match.group(1))


def parse_metadata_value(value: str) -> Any:
    parsed = parse_numeric(value)
    return parsed


def latest_puffer_log_config(run_dir: Path) -> dict[str, Any]:
    log_files = sorted(
        run_dir.glob("logs/**/*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not log_files:
        return {}

    try:
        payload = json.loads(log_files[-1].read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}

    env = payload.get("env", {}) if isinstance(payload.get("env"), dict) else {}
    train = payload.get("train", {}) if isinstance(payload.get("train"), dict) else {}
    policy = payload.get("policy", {}) if isinstance(payload.get("policy"), dict) else {}

    return {
        "seed": payload.get("seed", train.get("seed")),
        "timesteps": train.get("total_timesteps"),
        "target_dist": env.get("hover_target_dist"),
        "action_scale": env.get("action_scale"),
        "domain_randomization": env.get("domain_randomization"),
        "reset_yaw_range": env.get("reset_yaw_range"),
        "reset_vel_max": env.get("reset_vel_max"),
        "reset_pos_scale": env.get("reset_pos_scale"),
        "policy_num_layers": policy.get("num_layers"),
        "checkpoint_interval": payload.get("checkpoint_interval"),
        "checkpoint_dir": payload.get("checkpoint_dir"),
        "log_dir": payload.get("log_dir"),
        "tag": payload.get("tag"),
    }


def command_from_log_config(config: dict[str, Any]) -> str:
    required = [
        "tag",
        "checkpoint_dir",
        "log_dir",
        "checkpoint_interval",
        "seed",
        "timesteps",
        "target_dist",
        "domain_randomization",
        "action_scale",
        "reset_yaw_range",
        "reset_vel_max",
        "reset_pos_scale",
        "policy_num_layers",
    ]
    if any(config.get(key) is None for key in required):
        return ""

    return (
        "CUDA_VISIBLE_DEVICES=0 puffer train drone "
        f"--tag {config['tag']} "
        f"--checkpoint-dir {config['checkpoint_dir']} "
        f"--log-dir {config['log_dir']} "
        f"--checkpoint-interval {config['checkpoint_interval']} "
        f"--seed {config['seed']} "
        f"--train.seed {config['seed']} "
        f"--train.total-timesteps {config['timesteps']} "
        f"--env.hover-target-dist {config['target_dist']} "
        f"--env.domain-randomization {config['domain_randomization']} "
        f"--env.action-scale {config['action_scale']} "
        f"--env.reset-yaw-range {config['reset_yaw_range']} "
        f"--env.reset-vel-max {config['reset_vel_max']} "
        f"--env.reset-pos-scale {config['reset_pos_scale']} "
        f"--policy.num-layers {config['policy_num_layers']}"
    )


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
                metadata[key] = parse_metadata_value(value)

    log_config = latest_puffer_log_config(run_dir)
    for key, value in log_config.items():
        if value is not None and key not in metadata:
            metadata[key] = value

    command = ""
    command_path = run_dir / "command.sh"
    if command_path.exists():
        command = command_path.read_text(encoding="utf-8", errors="replace").strip()
    if not command:
        command = command_from_log_config(log_config)

    checkpoints = checkpoint_files(run_dir)
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "metadata": metadata,
        "command": command,
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics", {})
            summary = row.get("summary", {})
            losses = row.get("losses", {})
            metadata = row.get("metadata", {})
            latest = row.get("latest_checkpoint") or {}
            writer.writerow({
                "run_name": row.get("run_name"),
                "policy": losses.get("policy"),
                "value": losses.get("value"),
                "entropy": losses.get("entropy"),
                "total": losses.get("total"),
                "old_kl": losses.get("old_kl"),
                "kl": losses.get("kl"),
                "clipfrac": losses.get("clipfrac"),
                "steps": summary.get("steps"),
                "epoch": summary.get("epoch"),
                "timesteps": metadata.get("timesteps"),
                "seed": metadata.get("seed"),
                "target_dist": metadata.get("target_dist"),
                "action_scale": metadata.get("action_scale"),
                "domain_randomization": metadata.get("domain_randomization"),
                "reset_yaw_range": metadata.get("reset_yaw_range"),
                "reset_vel_max": metadata.get("reset_vel_max"),
                "reset_pos_scale": metadata.get("reset_pos_scale"),
                "exit_code": metadata.get("exit_code"),
                "checkpoint_count": row.get("checkpoint_count"),
                "latest_checkpoint_path": latest.get("path"),
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
                "mean_abs_action_raw": metrics.get("mean_abs_action_raw"),
                "max_abs_action_raw": metrics.get("max_abs_action_raw"),
                "raw_action_clip_frac": metrics.get("raw_action_clip_frac"),
                "mean_abs_action_clipped": metrics.get("mean_abs_action_clipped"),
                "max_abs_action_clipped": metrics.get("max_abs_action_clipped"),
                "clipped_action_saturation_frac": metrics.get("clipped_action_saturation_frac"),
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
                "r_terminal": metrics.get("r_terminal"),
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
