#!/usr/bin/env python3
"""Static Matrice 4D V0 alignment checks for ocean/drone.

This script intentionally avoids importing the Python standalone simulator. It
parses the C macros in dronelib.h and verifies that the PufferLib environment
itself still matches the CAD-aligned V0 motor geometry and allocation matrix.
"""

from __future__ import annotations

import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DRONELIB = ROOT / "dronelib.h"
RENDER = ROOT / "render.h"
BINDING = ROOT / "binding.c"
DRONE_H = ROOT / "drone.h"
TASKS_H = ROOT / "tasks.h"
CONFIG = ROOT.parents[1] / "config" / "drone.ini"


def read_macros(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    macros: dict[str, float] = {}
    for name, value in re.findall(r"^\s*#define\s+([A-Z0-9_]+)\s+([-+0-9.eEfF]+)", text, flags=re.MULTILINE):
        macros[name] = float(value.rstrip("fF"))
    return macros


def solve_4x4(a: list[list[float]], b: list[float]) -> list[float]:
    mat = [row[:] + [rhs] for row, rhs in zip(a, b)]
    n = 4
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(mat[row][col]))
        if abs(mat[pivot][col]) < 1e-12:
            raise AssertionError("allocation matrix is singular")
        mat[col], mat[pivot] = mat[pivot], mat[col]
        div = mat[col][col]
        mat[col] = [x / div for x in mat[col]]
        for row in range(n):
            if row == col:
                continue
            f = mat[row][col]
            mat[row] = [x - f * y for x, y in zip(mat[row], mat[col])]
    return [mat[i][-1] for i in range(n)]


def mat_vec(a: list[list[float]], x: list[float]) -> list[float]:
    return [sum(ai * xi for ai, xi in zip(row, x)) for row in a]


def close(actual: float, expected: float, tol: float, label: str) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: got {actual}, expected {expected} +/- {tol}")


def main() -> int:
    m = read_macros(DRONELIB)
    motor_order = ["FL", "FR", "RL", "RR"]
    motors = [
        (m["BASE_MOTOR_FL_X"], m["BASE_MOTOR_FL_Y"]),
        (m["BASE_MOTOR_FR_X"], m["BASE_MOTOR_FR_Y"]),
        (m["BASE_MOTOR_RL_X"], m["BASE_MOTOR_RL_Y"]),
        (m["BASE_MOTOR_RR_X"], m["BASE_MOTOR_RR_Y"]),
    ]
    yaw = [m["BASE_YAW_SIGN_FL"], m["BASE_YAW_SIGN_FR"], m["BASE_YAW_SIGN_RL"], m["BASE_YAW_SIGN_RR"]]

    close(abs(motors[1][0] - motors[0][0]), 0.3830, 1e-9, "FL-FR")
    close(abs(motors[3][0] - motors[2][0]), 0.3430, 1e-9, "RL-RR")
    close(abs(motors[0][1] - motors[2][1]), 0.3416, 1e-9, "front-rear")
    diag = math.dist(motors[0], motors[3])
    close(diag, 0.49845717970553904, 1e-9, "FL-RR diagonal")

    k_drag = m["BASE_K_DRAG"]
    allocation = [
        [1.0, 1.0, 1.0, 1.0],
        [motors[i][1] for i in range(4)],
        [-motors[i][0] for i in range(4)],
        [k_drag * yaw[i] for i in range(4)],
    ]

    hover_target = [m["BASE_MASS"] * m["BASE_GRAVITY"], 0.0, 0.0, 0.0]
    trim = solve_4x4(allocation, hover_target)
    achieved = mat_vec(allocation, trim)
    for i, (actual, expected) in enumerate(zip(achieved, hover_target)):
        close(actual, expected, 1e-9, f"hover allocation row {i}")

    hover_rpm = math.sqrt(trim[0] / m["BASE_K_THRUST"])
    close(hover_rpm, 5525.0, 1e-3, "hover RPM")

    max_thrust = m["BASE_K_THRUST"] * m["BASE_MAX_RPM"] * m["BASE_MAX_RPM"]
    yaw_actions = [0.10, -0.10, -0.10, 0.10]
    thrusts = [
        trim[i] + yaw_actions[i] * (max_thrust - trim[i]) if yaw_actions[i] >= 0.0
        else trim[i] + yaw_actions[i] * trim[i]
        for i in range(4)
    ]
    yaw_torque = mat_vec(allocation, thrusts)[3]
    if yaw_torque <= 0.0:
        raise AssertionError(f"positive yaw action pattern produced non-positive yaw torque: {yaw_torque}")

    render_text = RENDER.read_text(encoding="utf-8", errors="replace")
    if "resources/drone/matrice4d.glb" not in render_text:
        raise AssertionError("render.h does not prefer resources/drone/matrice4d.glb")
    if "agent->params.motor_x[0]" not in render_text:
        raise AssertionError("render primitive fallback is not using CAD motor positions")

    binding_text = BINDING.read_text(encoding="utf-8", errors="replace")
    drone_h_text = DRONE_H.read_text(encoding="utf-8", errors="replace")
    if "#define OBS_SIZE 23" not in binding_text or "observations + i*23" not in drone_h_text:
        raise AssertionError("OBS_SIZE and drone observation stride diverged")
    if drone_h_text.count("finalize_reset_potential(env, agent);") != 2:
        raise AssertionError("reset potential must be finalized after set_target in c_reset and c_step")

    tasks_text = TASKS_H.read_text(encoding="utf-8", errors="replace")
    for expected in [
        "agent->target->normal = (Vec3){0.0f, 0.0f, 1.0f};",
        "agent->target->orientation = (Quat){1.0f, 0.0f, 0.0f, 0.0f};",
        "agent->target->radius = 0.0f;",
    ]:
        if expected not in tasks_text:
            raise AssertionError("hover target metadata is incomplete")

    config_text = CONFIG.read_text(encoding="utf-8", errors="replace")
    if re.search(r"(?m)^num_layers\s*=\s*3\s*$", config_text) is None:
        raise AssertionError("config/drone.ini should use integer num_layers = 3")
    if re.search(r"(?m)^total_timesteps\s*=\s*3000000\s*$", config_text) is None:
        raise AssertionError("config/drone.ini should default to a 3M timestep smoke run")

    print("Matrice 4D V0 checks passed")
    print("motor_order:", motor_order)
    print("hover_trim_N:", [round(v, 6) for v in trim])
    print("hover_rpm:", round(hover_rpm, 3))
    print("positive_yaw_torque:", round(yaw_torque, 6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
