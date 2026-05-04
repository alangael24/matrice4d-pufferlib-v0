#!/usr/bin/env python3
import argparse
import ctypes
import ctypes.util
import json
from pathlib import Path

import numpy as np

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


def main():
    parser = argparse.ArgumentParser(description="CPU-vs-CUDA numeric parity check for drone env.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-agents", type=int, default=128)
    parser.add_argument("--action-amplitude", type=float, default=0.05)
    parser.add_argument("--obs-tol", type=float, default=1e-3)
    parser.add_argument("--reward-tol", type=float, default=1e-4)
    parser.add_argument("--out", default="")
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
    try:
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
    finally:
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
