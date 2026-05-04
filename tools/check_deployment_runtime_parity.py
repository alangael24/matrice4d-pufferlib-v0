#!/usr/bin/env python3
import argparse
import ctypes
import ctypes.util
import csv
import math
import struct
import subprocess
import tempfile
from pathlib import Path


OBS_SIZE = 23
NUM_ACTIONS = 4
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DECODER_OUTPUTS = NUM_ACTIONS + 1
DEFAULT_POLICY = Path(
    "/Users/alan/matrice4d_experiments/release_assets/v0.14-dr-medium-robust-baseline/"
    "v0_14_dr_medium_robust_seed44_scale0.7_reset32_latest.bin"
)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def load_expf():
    candidates = [None, ctypes.util.find_library("m")]
    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate) if candidate else ctypes.CDLL(None)
            expf = lib.expf
            expf.argtypes = [ctypes.c_float]
            expf.restype = ctypes.c_float
            return expf
        except (AttributeError, OSError):
            continue
    return None


EXPF = load_expf()


def c_expf(x):
    if EXPF is not None:
        return f32(EXPF(ctypes.c_float(f32(x))))
    return f32(math.exp(f32(x)))


def sigmoid(x):
    z = c_expf(-abs(x))
    if x >= 0.0:
        return f32(1.0 / f32(1.0 + z))
    return f32(z / f32(1.0 + z))


def expected_weights():
    return (
        HIDDEN_SIZE * OBS_SIZE
        + DECODER_OUTPUTS * HIDDEN_SIZE
        + NUM_ACTIONS
        + NUM_LAYERS * 3 * HIDDEN_SIZE * HIDDEN_SIZE
    )


def align8(value):
    return (value + 7) & ~7


def load_f32_file(path):
    data = Path(path).read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not a multiple of 4 bytes")
    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data))


class PythonDeploymentRuntime:
    def __init__(self, checkpoint, num_agents, reset_interval, action_scale):
        weights = load_f32_file(checkpoint)
        file_weights = len(weights)
        if file_weights != expected_weights():
            raise ValueError(f"checkpoint has {file_weights} floats; expected {expected_weights()}")
        weights = weights + [0.0] * 7
        self.num_agents = num_agents
        self.reset_interval = reset_interval
        self.action_scale = f32(action_scale)
        self.step = 0
        off = 0

        def take(count):
            nonlocal off
            if off + count > len(weights):
                raise ValueError("aligned checkpoint parser exceeded padded checkpoint size")
            out = weights[off : off + count]
            off += count
            off = align8(off)
            return out

        self.encoder = take(HIDDEN_SIZE * OBS_SIZE)
        self.decoder = take(DECODER_OUTPUTS * HIDDEN_SIZE)
        self.logstd = take(NUM_ACTIONS)
        self.mingru = [take(3 * HIDDEN_SIZE * HIDDEN_SIZE) for _ in range(NUM_LAYERS)]
        if off > file_weights + 7:
            raise ValueError("checkpoint parser exceeded expected aligned capacity")
        self.state = [[0.0] * (num_agents * HIDDEN_SIZE) for _ in range(NUM_LAYERS)]

    def reset_state(self):
        for layer in self.state:
            for i in range(len(layer)):
                layer[i] = 0.0

    def matmul(self, x, weights, in_dim, out_dim):
        out = [0.0] * (self.num_agents * out_dim)
        for b in range(self.num_agents):
            x_off = b * in_dim
            out_off = b * out_dim
            for o in range(out_dim):
                w_off = o * in_dim
                total = 0.0
                for i in range(in_dim):
                    total = f32(total + f32(x[x_off + i] * weights[w_off + i]))
                out[out_off + o] = total
        return out

    def forward(self, obs):
        if self.reset_interval > 0 and self.step % self.reset_interval == 0:
            self.reset_state()
        x = self.matmul(obs, self.encoder, OBS_SIZE, HIDDEN_SIZE)
        for layer_idx, weight in enumerate(self.mingru):
            combined = self.matmul(x, weight, HIDDEN_SIZE, 3 * HIDDEN_SIZE)
            out = [0.0] * (self.num_agents * HIDDEN_SIZE)
            state_l = self.state[layer_idx]
            for b in range(self.num_agents):
                c_off = b * 3 * HIDDEN_SIZE
                h_off = b * HIDDEN_SIZE
                for h in range(HIDDEN_SIZE):
                    hidden = combined[c_off + h]
                    gate = combined[c_off + HIDDEN_SIZE + h]
                    proj = combined[c_off + 2 * HIDDEN_SIZE + h]
                    hidden_tilde = f32(hidden + 0.5) if hidden >= 0.0 else sigmoid(hidden)
                    gate_sigmoid = sigmoid(gate)
                    prev = state_l[h_off + h]
                    mingru_out = f32(prev + f32(gate_sigmoid * f32(hidden_tilde - prev)))
                    proj_sigmoid = sigmoid(proj)
                    out[h_off + h] = f32(
                        f32(proj_sigmoid * mingru_out)
                        + f32(f32(1.0 - proj_sigmoid) * x[h_off + h])
                    )
                    state_l[h_off + h] = mingru_out
            x = out
        decoder_out = self.matmul(x, self.decoder, HIDDEN_SIZE, DECODER_OUTPUTS)
        actions = []
        for b in range(self.num_agents):
            base = b * DECODER_OUTPUTS
            actions.extend(decoder_out[base : base + NUM_ACTIONS])
        self.step += 1
        return actions

    def scale_actions(self, raw):
        scaled = []
        for value in raw:
            clipped = min(max(value, -1.0), 1.0)
            scaled.append(f32(min(max(f32(clipped * self.action_scale), -1.0), 1.0)))
        return scaled


def make_fixture(steps, num_agents):
    obs = []
    for step in range(steps):
        for agent in range(num_agents):
            for idx in range(OBS_SIZE):
                v = 0.25 * math.sin(0.17 * step + 0.11 * agent + 0.07 * idx)
                v += 0.10 * math.cos(0.03 * step * (idx + 1))
                obs.append(f32(v))
    return obs


def write_obs(path, obs, num_agents):
    floats_per_step = num_agents * OBS_SIZE
    with Path(path).open("w", encoding="utf-8") as file:
        for i, value in enumerate(obs):
            file.write(f"{value:.17g}")
            if (i + 1) % floats_per_step == 0:
                file.write("\n")
            else:
                file.write(",")


def compile_dumper(repo, output):
    source = repo / "tools" / "dump_deployment_actions.c"
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "clang",
        "-O3",
        "-DNDEBUG",
        "-Wall",
        "-Werror=return-type",
        "-Wno-unused-function",
        "-Wno-unused-variable",
        "-I./tools",
        str(source),
        "-lm",
        "-o",
        str(output),
    ]
    subprocess.run(cmd, cwd=repo, check=True)


def run_c_dumper(repo, dumper, checkpoint, obs_path, steps, num_agents, reset_interval, action_scale):
    cmd = [
        str(dumper),
        str(checkpoint),
        str(obs_path),
        str(steps),
        str(num_agents),
        str(reset_interval),
        str(action_scale),
    ]
    proc = subprocess.run(cmd, cwd=repo, check=True, text=True, capture_output=True)
    rows = []
    for row in csv.DictReader(proc.stdout.splitlines()):
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare Python policy math against C deployment runtime.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_POLICY))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--action-scale", type=float, default=0.7)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dumper", default="")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    dumper = Path(args.dumper).resolve() if args.dumper else repo / "build" / "dump_deployment_actions"
    compile_dumper(repo, dumper)

    fixture = make_fixture(args.steps, args.num_agents)
    py_runtime = PythonDeploymentRuntime(
        checkpoint, args.num_agents, args.reset_interval, args.action_scale
    )
    py_raw_rows = []
    py_scaled_rows = []
    floats_per_step = args.num_agents * OBS_SIZE
    for step in range(args.steps):
        obs = fixture[step * floats_per_step : (step + 1) * floats_per_step]
        raw = py_runtime.forward(obs)
        scaled = py_runtime.scale_actions(raw)
        py_raw_rows.append(raw)
        py_scaled_rows.append(scaled)

    with tempfile.TemporaryDirectory(prefix="m4d_deploy_parity_") as tmp:
        obs_path = Path(tmp) / "obs.csv"
        write_obs(obs_path, fixture, args.num_agents)
        c_rows = run_c_dumper(
            repo,
            dumper,
            checkpoint,
            obs_path,
            args.steps,
            args.num_agents,
            args.reset_interval,
            args.action_scale,
        )

    expected_rows = args.steps * args.num_agents
    if len(c_rows) != expected_rows:
        raise RuntimeError(f"C dumper returned {len(c_rows)} rows; expected {expected_rows}")

    max_raw_diff = 0.0
    max_scaled_diff = 0.0
    worst = None
    for row_index, row in enumerate(c_rows):
        step = int(row["step"])
        agent = int(row["agent"])
        raw_base = agent * NUM_ACTIONS
        py_raw = py_raw_rows[step][raw_base : raw_base + NUM_ACTIONS]
        py_scaled = py_scaled_rows[step][raw_base : raw_base + NUM_ACTIONS]
        c_raw = [float(row[f"raw{i}"]) for i in range(NUM_ACTIONS)]
        c_scaled = [float(row[f"scaled{i}"]) for i in range(NUM_ACTIONS)]
        for i in range(NUM_ACTIONS):
            raw_diff = abs(py_raw[i] - c_raw[i])
            scaled_diff = abs(py_scaled[i] - c_scaled[i])
            if raw_diff > max_raw_diff:
                max_raw_diff = raw_diff
                worst = ("raw", step, agent, i, py_raw[i], c_raw[i])
            if scaled_diff > max_scaled_diff:
                max_scaled_diff = scaled_diff
                worst = ("scaled", step, agent, i, py_scaled[i], c_scaled[i])

    passed = max(max_raw_diff, max_scaled_diff) <= args.tolerance
    print(f"checkpoint={checkpoint}")
    print(f"rows={expected_rows} steps={args.steps} num_agents={args.num_agents}")
    print(f"reset_interval={args.reset_interval} action_scale={args.action_scale}")
    print(f"max_raw_abs_diff={max_raw_diff:.9g}")
    print(f"max_scaled_abs_diff={max_scaled_diff:.9g}")
    print(f"tolerance={args.tolerance:.9g}")
    print(f"passed={int(passed)}")
    if not passed:
        print(f"worst={worst}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
