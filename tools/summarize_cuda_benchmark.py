#!/usr/bin/env python3
"""Summarize Matrice 4D CUDA env benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_exit_code(path: Path) -> int | None:
    try:
        return int(read_text(path).strip())
    except ValueError:
        return None


def read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in read_text(path).splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value
    return out


def match_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def match_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_v0_checks(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "path": str(path),
        "exit_code": read_exit_code(Path(str(path) + ".exit_code")),
        "passed": "Matrice 4D V0 checks passed" in text,
        "hover_rpm": match_float(text, r"hover_rpm:\s*([0-9.]+)"),
    }


def parse_cuda_checks(path: Path) -> dict[str, Any]:
    text = read_text(path)
    payload = extract_json_object(text) or {}
    return {
        "path": str(path),
        "exit_code": read_exit_code(Path(str(path) + ".exit_code")),
        "passed": bool(payload.get("passed")),
        "report": payload,
    }


def parse_envspeed(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "path": str(path),
        "exit_code": read_exit_code(Path(str(path) + ".exit_code")),
        "total_agents": match_int(text, r"total_agents=(\d+)"),
        "buffers": match_int(text, r"buffers=(\d+)"),
        "threads": match_int(text, r"threads=(\d+)"),
        "horizon": match_int(text, r"horizon=(\d+)"),
        "num_envs": match_int(text, r"num_envs=(\d+)"),
        "obs_size": match_int(text, r"obs_size=(\d+)"),
        "num_atns": match_int(text, r"num_atns=(\d+)"),
        "rollout_ms": match_float(text, r"rollout time:\s*([0-9.]+)\s*ms"),
        "rollout_steps": match_int(text, r"rollout time:\s*[0-9.]+\s*ms\s*\((\d+)\s*steps\)"),
        "eval_gpu_ms": match_float(text, r"eval_gpu:\s*([0-9.]+)\s*ms/rollout"),
        "eval_env_ms": match_float(text, r"eval_env:\s*([0-9.]+)\s*ms/rollout"),
        "throughput_msteps_s": match_float(text, r"throughput:\s*([0-9.]+)\s*M steps/s"),
    }


def load_train_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"run_count": 0, "runs": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"run_count": 0, "runs": []}


def selected_train_rows(train_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in train_summary.get("runs", []):
        metadata = run.get("metadata", {})
        summary = run.get("summary", {})
        metrics = run.get("metrics", {})
        rows.append({
            "run_name": run.get("run_name"),
            "backend": metadata.get("backend"),
            "exit_code": metadata.get("exit_code"),
            "steps": summary.get("steps"),
            "sps": summary.get("sps"),
            "perf": metrics.get("perf"),
            "score": metrics.get("score"),
            "episode_return": metrics.get("episode_return"),
            "ema_dist": metrics.get("ema_dist"),
            "ema_vel": metrics.get("ema_vel"),
            "ema_omega": metrics.get("ema_omega"),
            "oob": metrics.get("oob"),
            "timeout": metrics.get("timeout"),
            "latest_checkpoint": (run.get("latest_checkpoint") or {}).get("path"),
        })
    return rows


def speed_delta(cpu: dict[str, Any], cuda: dict[str, Any]) -> dict[str, Any]:
    cpu_t = cpu.get("throughput_msteps_s")
    cuda_t = cuda.get("throughput_msteps_s")
    cpu_env = cpu.get("eval_env_ms")
    cuda_env = cuda.get("eval_env_ms")
    return {
        "throughput_speedup": (cuda_t / cpu_t) if cpu_t and cuda_t else None,
        "eval_env_ms_delta": (cuda_env - cpu_env) if cpu_env is not None and cuda_env is not None else None,
        "cuda_faster": bool(cpu_t and cuda_t and cuda_t > cpu_t),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    correctness = payload["correctness"]
    speed = payload["speed"]
    train_rows = payload["training"]["runs"]

    def check_row(label: str, key: str, detail: str) -> str:
        row = correctness.get(key, {})
        return f"| {label} | {row.get('passed')} | {detail} |"

    lines = [
        "# CUDA Env Benchmark",
        "",
        "## Correctness",
        "",
        "| Check | Passed | Detail |",
        "| --- | ---: | --- |",
        f"| Matrice 4D V0 | {correctness['v0_checks'].get('passed')} | hover_rpm={correctness['v0_checks'].get('hover_rpm')} |",
        check_row("CPU vs CUDA zero-action 1000", "cuda_zero_action_1000", "required; see correctness/cuda_zero_action_1000.txt"),
        check_row("CPU vs CUDA amp=0.05 200", "cuda_amp005_200", "required; see correctness/cuda_amp005_200.txt"),
        check_row("CPU vs CUDA amp=0.05 1000", "cuda_amp005_1000_diagnostic", "non-blocking diagnostic; see correctness/cuda_amp005_1000_diagnostic.txt"),
        "",
        "## Speed",
        "",
        "| Backend | Throughput M steps/s | Rollout ms | eval_gpu ms | eval_env ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("cpu_env", "cuda_env"):
        row = speed[name]
        lines.append(
            f"| {name} | {row.get('throughput_msteps_s')} | {row.get('rollout_ms')} | "
            f"{row.get('eval_gpu_ms')} | {row.get('eval_env_ms')} |"
        )
    lines += [
        "",
        f"Speedup: {speed['delta'].get('throughput_speedup')}",
        "",
        "## Training",
        "",
        "| Run | Backend | Exit | Steps | SPS | Perf | Score | Return | ema_dist | ema_vel | ema_omega | oob | timeout |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in train_rows:
        lines.append(
            f"| {row.get('run_name')} | {row.get('backend')} | {row.get('exit_code')} | "
            f"{row.get('steps')} | {row.get('sps')} | {row.get('perf')} | {row.get('score')} | "
            f"{row.get('episode_return')} | {row.get('ema_dist')} | {row.get('ema_vel')} | "
            f"{row.get('ema_omega')} | {row.get('oob')} | {row.get('timeout')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    bench = args.bench_root
    train_summary = load_train_summary(bench / "train_summary.json")
    cpu_speed = parse_envspeed(bench / "speed" / "envspeed_cpu_env.txt")
    cuda_speed = parse_envspeed(bench / "speed" / "envspeed_cuda_env.txt")
    smoke_path = bench / "correctness" / "cuda_amp005_200.txt"
    legacy_smoke_path = bench / "correctness" / "cuda_checks.txt"
    payload = {
        "bench_root": str(bench),
        "metadata": read_env(bench / "benchmark_metadata.env"),
        "git_commit": read_text(bench / "git_commit.txt").strip(),
        "correctness": {
            "v0_checks": parse_v0_checks(bench / "correctness" / "matrice4d_v0_checks.txt"),
            "cuda_checks": parse_cuda_checks(smoke_path if smoke_path.exists() else legacy_smoke_path),
            "cuda_zero_action_1000": parse_cuda_checks(bench / "correctness" / "cuda_zero_action_1000.txt"),
            "cuda_amp005_200": parse_cuda_checks(smoke_path if smoke_path.exists() else legacy_smoke_path),
            "cuda_amp005_1000_diagnostic": parse_cuda_checks(bench / "correctness" / "cuda_amp005_1000_diagnostic.txt"),
        },
        "speed": {
            "cpu_env": cpu_speed,
            "cuda_env": cuda_speed,
            "delta": speed_delta(cpu_speed, cuda_speed),
        },
        "training": {
            "run_count": train_summary.get("run_count", 0),
            "runs": selected_train_rows(train_summary),
            "raw_summary_path": str(bench / "train_summary.json"),
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown, payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
