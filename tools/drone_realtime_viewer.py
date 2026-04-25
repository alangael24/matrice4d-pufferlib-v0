import argparse
import ctypes
import os
import time
from pathlib import Path

import numpy as np
import torch

from pufferlib import _C


OBS_SIZE = 23
NUM_ACTIONS = 4
DEFAULT_POLICY = Path(
    "/mnt/c/Users/alang/matrice4d_experiments/runs/"
    "m4d_v0_2026-04-23_ppo_hover_30M_seed000/policies/"
    "run2_easy_curriculum_30M_target05_scale02_seed042_final.bin"
)


def tensor_from_ptr(ptr, shape):
    n = int(np.prod(shape))
    buf = (ctypes.c_float * n).from_address(ptr)
    return torch.frombuffer(buf, dtype=torch.float32).reshape(shape)


def native_sigmoid(x):
    z = torch.exp(-torch.abs(x))
    return torch.where(x >= 0.0, 1.0 / (1.0 + z), z / (1.0 + z))


class NativeMinGRUPolicy(torch.nn.Module):
    def __init__(self, checkpoint, obs_size=OBS_SIZE, hidden_size=128, num_layers=3, num_actions=NUM_ACTIONS):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_actions = num_actions

        weights = np.fromfile(checkpoint, dtype=np.float32)
        offset = 0

        def take(shape):
            nonlocal offset
            count = int(np.prod(shape))
            if offset + count > weights.size:
                raise ValueError(f"Checkpoint ended early at {offset}; need {count} more floats")
            out = torch.from_numpy(weights[offset : offset + count].copy()).reshape(shape)
            offset += count
            return out

        self.encoder_weight = torch.nn.Parameter(take((hidden_size, obs_size)), requires_grad=False)
        self.decoder_weight = torch.nn.Parameter(take((num_actions + 1, hidden_size)), requires_grad=False)
        self.logstd = torch.nn.Parameter(take((1, num_actions)), requires_grad=False)
        self.mingru_weights = torch.nn.ParameterList(
            [torch.nn.Parameter(take((3 * hidden_size, hidden_size)), requires_grad=False) for _ in range(num_layers)]
        )

        if offset != weights.size:
            raise ValueError(f"Checkpoint has {weights.size - offset} unused floats")

    def initial_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    @torch.no_grad()
    def forward(self, obs, state):
        x = obs.reshape(obs.shape[0], -1).float() @ self.encoder_weight.T
        next_layers = []

        for layer, weight in enumerate(self.mingru_weights):
            combined = x @ weight.T
            hidden, gate, proj = combined.chunk(3, dim=1)
            hidden_tilde = torch.where(hidden >= 0.0, hidden + 0.5, native_sigmoid(hidden))
            gate_sigmoid = native_sigmoid(gate)
            mingru_out = state[layer] + gate_sigmoid * (hidden_tilde - state[layer])
            proj_sigmoid = native_sigmoid(proj)
            x = proj_sigmoid * mingru_out + (1.0 - proj_sigmoid) * x
            next_layers.append(mingru_out)

        decoder_out = x @ self.decoder_weight.T
        actions = decoder_out[:, : self.num_actions]
        return actions.contiguous(), torch.stack(next_layers, dim=0)


def make_args(
    num_drones,
    hover_target_dist,
    action_scale,
    domain_randomization,
    reset_pos_scale=1.0,
    reset_yaw_range=0.0,
    reset_vel_max=0.0,
):
    return {
        "env_name": "drone",
        "vec": {
            "total_agents": num_drones,
            "num_buffers": 1,
            "num_threads": 1,
        },
        "env": {
            "task": 1,
            "num_drones": num_drones,
            "max_rings": 10,
            "alpha_dist": 0.782192,
            "alpha_hover": 0.071445,
            "alpha_omega": 0.00135588,
            "alpha_shaping": 3.9754,
            "hover_target_dist": hover_target_dist,
            "hover_dist": 0.1,
            "hover_omega": 0.1,
            "hover_vel": 0.1,
            "domain_randomization": domain_randomization,
            "action_scale": action_scale,
            "reset_pos_scale": reset_pos_scale,
            "reset_yaw_range": reset_yaw_range,
            "reset_vel_max": reset_vel_max,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Realtime local viewer for the Matrice 4D Run 2 native checkpoint.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_POLICY))
    parser.add_argument("--num-drones", type=int, default=1)
    parser.add_argument("--hover-target-dist", type=float, default=0.5)
    parser.add_argument("--domain-randomization", type=float, default=0.0)
    parser.add_argument("--action-scale", type=float, default=0.2)
    parser.add_argument("--reset-pos-scale", type=float, default=1.0)
    parser.add_argument("--reset-yaw-range", type=float, default=0.0)
    parser.add_argument("--reset-vel-max", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-elevation", type=float, default=0.35)
    parser.add_argument("--model-scale", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=0, help="Stop after N steps. 0 means run until ESC/Ctrl-C.")
    parser.add_argument("--no-render", action="store_true", help="Step without opening the Raylib window.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
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
    terminals = tensor_from_ptr(vec.terminals_ptr, (vec.total_agents,))

    os.environ.setdefault("PUFFER_DRONE_CAMERA_DISTANCE", str(args.camera_distance))
    os.environ.setdefault("PUFFER_DRONE_CAMERA_ELEVATION", str(args.camera_elevation))
    os.environ.setdefault("PUFFER_DRONE_MODEL_SCALE", str(args.model_scale))
    os.environ.setdefault("PUFFER_DRONE_INSPECT", "1")
    os.environ.setdefault("PUFFER_DRONE_FOLLOW", "1")

    vec.reset()
    frame_dt = 1.0 / args.fps if args.fps > 0 else 0.0

    try:
        step = 0
        while args.steps <= 0 or step < args.steps:
            started = time.perf_counter()
            if not args.no_render:
                vec.render(0)

            actions, state = policy(obs, state)
            vec.cpu_step(actions.data_ptr())

            done = terminals > 0.0
            if torch.any(done):
                state[:, done, :] = 0.0

            if frame_dt > 0:
                elapsed = time.perf_counter() - started
                if elapsed < frame_dt:
                    time.sleep(frame_dt - elapsed)

            step += 1
    finally:
        vec.close()


if __name__ == "__main__":
    main()
