#!/usr/bin/env python3
import argparse
import csv
import ctypes
import ctypes.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pufferlib import _C
from tools.drone_realtime_viewer import tensor_from_ptr


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


def load_cudart():
    path = ctypes.util.find_library("cudart") or "libcudart.so"
    lib = ctypes.CDLL(path)
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cudaDeviceSynchronize.argtypes = []
    lib.cudaDeviceSynchronize.restype = ctypes.c_int
    lib.cudaGetErrorString.argtypes = [ctypes.c_int]
    lib.cudaGetErrorString.restype = ctypes.c_char_p
    return lib


def cuda_check(lib, code, expr):
    if code != 0:
        msg = lib.cudaGetErrorString(code).decode("utf-8", errors="replace")
        raise RuntimeError(f"{expr} failed: {msg}")


def make_args(num_agents):
    return {
        "env_name": "drone",
        "vec": {
            "total_agents": num_agents,
            "num_buffers": 1,
            "num_threads": 1,
            "seed": 123,
        },
        "env": {
            "task": 1,
            "num_drones": num_agents,
            "max_rings": 10,
            "alpha_dist": 0.782192,
            "alpha_hover": 0.071445,
            "alpha_shaping": 3.9754,
            "alpha_omega": 0.00135588,
            "alpha_omega_xy": 0.00135588,
            "alpha_omega_z": 0.00135588,
            "alpha_omega_z_sq": 0.0025,
            "alpha_omega_z_mult": 5.0,
            "hover_target_dist": 0.0,
            "oob_radius": 12.0,
            "hover_dist": 0.1,
            "hover_omega": 0.1,
            "hover_vel": 0.1,
            "domain_randomization": 0.0,
            "dr_mass": 0.0,
            "dr_inertia": 0.0,
            "dr_k_thrust": 0.0,
            "dr_linear_drag": 0.0,
            "dr_yaw_drag": 0.0,
            "dr_motor_lag": 0.0,
            "dr_com_xy": 0.0,
            "dr_com_z": 0.0,
            "action_scale": 1.0,
            "reset_pos_scale": 0.0,
            "reset_yaw_range": 0.0,
            "reset_vel_max": 0.0,
            "action_latency": 0.0,
            "sensor_noise": 0.0,
        },
    }


def copy_device_to_numpy(lib, device_ptr, shape):
    out = np.empty(shape, dtype=np.float32)
    cuda_check(
        lib,
        lib.cudaMemcpy(
            ctypes.c_void_p(out.ctypes.data),
            ctypes.c_void_p(device_ptr),
            out.nbytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        ),
        "cudaMemcpy(D2H)",
    )
    return out


def copy_numpy_to_device(lib, array, device_ptr):
    arr = np.ascontiguousarray(array, dtype=np.float32)
    cuda_check(
        lib,
        lib.cudaMemcpy(
            ctypes.c_void_p(device_ptr),
            ctypes.c_void_p(arr.ctypes.data),
            arr.nbytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        ),
        "cudaMemcpy(H2D)",
    )


def make_actions(step, num_agents, amplitude):
    idx = np.arange(num_agents * 4, dtype=np.float32).reshape(num_agents, 4)
    return amplitude * np.sin(0.013 * idx + 0.017 * float(step)).astype(np.float32)


STATE_FIELDS = {
    "pos": ("x", "y", "z"),
    "vel": ("x", "y", "z"),
    "quat": ("w", "x", "y", "z"),
    "omega": ("x", "y", "z"),
    "rpms": ("FL", "FR", "RL", "RR"),
    "target_pos": ("x", "y", "z"),
    "target_normal": ("x", "y", "z"),
    "prev_pos": ("x", "y", "z"),
    "motor_x": ("FL", "FR", "RL", "RR"),
    "motor_y": ("FL", "FR", "RL", "RR"),
    "yaw_sign": ("FL", "FR", "RL", "RR"),
    "hover_trim": ("FL", "FR", "RL", "RR"),
}

STATE_SCALARS = (
    "prev_potential",
    "episode_return",
    "episode_length",
    "mass",
    "ixx",
    "iyy",
    "izz",
    "k_thrust",
    "k_drag",
    "b_drag",
    "k_mot",
    "action_scale",
)


def csv_fieldnames(include_state):
    fields = [
        "step",
        "max_obs_diff",
        "argmax_agent",
        "argmax_obs_index",
        "obs_cpu_argmax",
        "obs_cuda_argmax",
        "reward_cpu",
        "reward_cuda",
        "reward_diff",
        "terminal_cpu",
        "terminal_cuda",
        "terminal_diff",
        "max_reward_diff",
        "max_terminal_diff",
    ]
    if include_state:
        for name, suffixes in STATE_FIELDS.items():
            for suffix in suffixes:
                fields += [
                    f"{name}_{suffix}_cpu",
                    f"{name}_{suffix}_cuda",
                    f"{name}_{suffix}_diff",
                ]
        for name in STATE_SCALARS:
            fields += [f"{name}_cpu", f"{name}_cuda", f"{name}_diff"]
    return fields


def add_state_columns(row, cpu_state, cuda_state):
    for name, suffixes in STATE_FIELDS.items():
        cpu_values = cpu_state[name]
        cuda_values = cuda_state[name]
        for idx, suffix in enumerate(suffixes):
            cpu_val = float(cpu_values[idx])
            cuda_val = float(cuda_values[idx])
            row[f"{name}_{suffix}_cpu"] = cpu_val
            row[f"{name}_{suffix}_cuda"] = cuda_val
            row[f"{name}_{suffix}_diff"] = abs(cpu_val - cuda_val)
    for name in STATE_SCALARS:
        cpu_val = float(cpu_state[name])
        cuda_val = float(cuda_state[name])
        row[f"{name}_cpu"] = cpu_val
        row[f"{name}_cuda"] = cuda_val
        row[f"{name}_diff"] = abs(cpu_val - cuda_val)


def make_diff_row(step, cpu_obs, gpu_obs, cpu_rewards, gpu_rewards,
                  cpu_terms, gpu_terms, cpu_vec, gpu_vec, include_state):
    obs_abs = np.abs(cpu_obs - gpu_obs)
    flat_idx = int(np.argmax(obs_abs))
    agent_idx, obs_idx = np.unravel_index(flat_idx, obs_abs.shape)
    reward_diff = float(abs(cpu_rewards[agent_idx] - gpu_rewards[agent_idx]))
    terminal_diff = float(abs(cpu_terms[agent_idx] - gpu_terms[agent_idx]))
    row = {
        "step": step,
        "max_obs_diff": float(obs_abs[agent_idx, obs_idx]),
        "argmax_agent": int(agent_idx),
        "argmax_obs_index": int(obs_idx),
        "obs_cpu_argmax": float(cpu_obs[agent_idx, obs_idx]),
        "obs_cuda_argmax": float(gpu_obs[agent_idx, obs_idx]),
        "reward_cpu": float(cpu_rewards[agent_idx]),
        "reward_cuda": float(gpu_rewards[agent_idx]),
        "reward_diff": reward_diff,
        "terminal_cpu": float(cpu_terms[agent_idx]),
        "terminal_cuda": float(gpu_terms[agent_idx]),
        "terminal_diff": terminal_diff,
        "max_reward_diff": float(np.max(np.abs(cpu_rewards - gpu_rewards))),
        "max_terminal_diff": float(np.max(np.abs(cpu_terms - gpu_terms))),
    }
    if include_state:
        add_state_columns(row, cpu_vec.debug_state(int(agent_idx)), gpu_vec.debug_state(int(agent_idx)))
    return row


def main():
    parser = argparse.ArgumentParser(description="CPU-vs-CUDA numeric parity check for drone env.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-agents", type=int, default=128)
    parser.add_argument("--action-amplitude", type=float, default=0.05)
    parser.add_argument("--obs-tol", type=float, default=1e-3)
    parser.add_argument("--reward-tol", type=float, default=1e-4)
    parser.add_argument("--out", default="")
    parser.add_argument("--dump-diffs", default="", help="Optional CSV path with per-step max diffs.")
    parser.add_argument("--dump-state", action="store_true",
                        help="Include CPU/CUDA physical state for the max-drift agent in --dump-diffs.")
    args = parser.parse_args()

    if getattr(_C, "env_name", None) != "drone":
        raise RuntimeError(f"pufferlib._C is built for {_C.env_name}, not drone")
    if int(getattr(_C, "gpu", 0)) != 1:
        raise RuntimeError("pufferlib._C is not the CUDA build. Run: bash build.sh drone")

    cudart = load_cudart()
    cfg = make_args(args.num_agents)
    cpu_vec = _C.create_vec(cfg, 0)
    gpu_vec = _C.create_vec(cfg, 1)
    d_actions = ctypes.c_void_p()
    action_bytes = args.num_agents * 4 * np.dtype(np.float32).itemsize
    cuda_check(cudart, cudart.cudaMalloc(ctypes.byref(d_actions), action_bytes), "cudaMalloc(actions)")

    max_obs = 0.0
    max_reward = 0.0
    max_terminal = 0.0
    first_terminal_mismatch = None
    csv_file = None
    writer = None
    try:
        if args.dump_state and not args.dump_diffs:
            raise RuntimeError("--dump-state requires --dump-diffs")
        if args.dump_diffs:
            dump_path = Path(args.dump_diffs)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = dump_path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames(args.dump_state))
            writer.writeheader()

        cpu_vec.reset()
        gpu_vec.reset()
        cuda_check(cudart, cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize(reset)")

        cpu_obs = tensor_from_ptr(cpu_vec.obs_ptr, (cpu_vec.total_agents, cpu_vec.obs_size)).numpy()
        cpu_rewards = tensor_from_ptr(cpu_vec.rewards_ptr, (cpu_vec.total_agents,)).numpy()
        cpu_terms = tensor_from_ptr(cpu_vec.terminals_ptr, (cpu_vec.total_agents,)).numpy()

        gpu_obs = copy_device_to_numpy(cudart, gpu_vec.gpu_obs_ptr, cpu_obs.shape)
        gpu_rewards = copy_device_to_numpy(cudart, gpu_vec.gpu_rewards_ptr, cpu_rewards.shape)
        gpu_terms = copy_device_to_numpy(cudart, gpu_vec.gpu_terminals_ptr, cpu_terms.shape)
        max_obs = max(max_obs, float(np.max(np.abs(cpu_obs - gpu_obs))))
        max_reward = max(max_reward, float(np.max(np.abs(cpu_rewards - gpu_rewards))))
        max_terminal = max(max_terminal, float(np.max(np.abs(cpu_terms - gpu_terms))))
        if writer is not None:
            writer.writerow(make_diff_row(-1, cpu_obs, gpu_obs, cpu_rewards, gpu_rewards,
                                          cpu_terms, gpu_terms, cpu_vec, gpu_vec,
                                          args.dump_state))

        for step in range(args.steps):
            actions = make_actions(step, args.num_agents, args.action_amplitude)
            cpu_vec.cpu_step(actions.ctypes.data)
            copy_numpy_to_device(cudart, actions, d_actions.value)
            gpu_vec.gpu_step(d_actions.value)
            cuda_check(cudart, cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize(step)")

            gpu_obs = copy_device_to_numpy(cudart, gpu_vec.gpu_obs_ptr, cpu_obs.shape)
            gpu_rewards = copy_device_to_numpy(cudart, gpu_vec.gpu_rewards_ptr, cpu_rewards.shape)
            gpu_terms = copy_device_to_numpy(cudart, gpu_vec.gpu_terminals_ptr, cpu_terms.shape)

            obs_diff = float(np.max(np.abs(cpu_obs - gpu_obs)))
            reward_diff = float(np.max(np.abs(cpu_rewards - gpu_rewards)))
            terminal_diff = float(np.max(np.abs(cpu_terms - gpu_terms)))
            max_obs = max(max_obs, obs_diff)
            max_reward = max(max_reward, reward_diff)
            max_terminal = max(max_terminal, terminal_diff)
            if terminal_diff > 0.0 and first_terminal_mismatch is None:
                first_terminal_mismatch = step
            if writer is not None:
                writer.writerow(make_diff_row(step, cpu_obs, gpu_obs, cpu_rewards, gpu_rewards,
                                              cpu_terms, gpu_terms, cpu_vec, gpu_vec,
                                              args.dump_state))
    finally:
        if csv_file is not None:
            csv_file.close()
        cudart.cudaFree(d_actions)
        cpu_vec.close()
        gpu_vec.close()

    report = {
        "steps": args.steps,
        "num_agents": args.num_agents,
        "action_amplitude": args.action_amplitude,
        "max_abs_obs": max_obs,
        "max_abs_reward": max_reward,
        "max_abs_terminal": max_terminal,
        "first_terminal_mismatch": first_terminal_mismatch,
        "obs_tol": args.obs_tol,
        "reward_tol": args.reward_tol,
        "passed": max_obs < args.obs_tol and max_reward < args.reward_tol and max_terminal == 0.0,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
