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
    match = re.search(rf"(?:^|\s){re.escape(key)}\s+({token})(?=\s|$)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1) if word else parse_numeric(match.group(1))


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
    command_path = run_dir / "command.sh"
    if command_path.exists():
        command = command_path.read_text(encoding="utf-8", errors="replace").strip()

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
    fieldnames = [
        "run_name",
        "exit_code",
        "steps",
        "perf",
        "score",
        "oob",
        "timeout",
        "episode_return",
        "episode_length",
        "ema_dist",
        "ema_vel",
        "ema_omega",
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
            latest = row.get("latest_checkpoint") or {}
            writer.writerow({
                "run_name": row.get("run_name"),
                "exit_code": metadata.get("exit_code"),
                "steps": summary.get("steps"),
                "perf": metrics.get("perf"),
                "score": metrics.get("score"),
                "oob": metrics.get("oob"),
                "timeout": metrics.get("timeout"),
                "episode_return": metrics.get("episode_return"),
                "episode_length": metrics.get("episode_length"),
                "ema_dist": metrics.get("ema_dist"),
                "ema_vel": metrics.get("ema_vel"),
                "ema_omega": metrics.get("ema_omega"),
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
