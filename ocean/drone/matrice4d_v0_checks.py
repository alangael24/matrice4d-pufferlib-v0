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
    obs_macro_ok = "#define OBS_SIZE DRONE_OBS_SIZE" in binding_text or "#define OBS_SIZE 23" in binding_text
    stride_ok = "observations + i * DRONE_OBS_SIZE" in drone_h_text or "observations + i*23" in drone_h_text
    if not obs_macro_ok or not stride_ok:
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
    if re.search(r"(?m)^hover_target_dist\s*=\s*5\.0\s*$", config_text) is None:
        raise AssertionError("config/drone.ini should use float hover_target_dist = 5.0")
    if re.search(r"(?m)^oob_radius\s*=\s*12\.0\s*$", config_text) is None:
        raise AssertionError("config/drone.ini should expose oob_radius = 12.0 for target 5 m")
    if re.search(r"(?m)^num_layers\s*=\s*3\s*$", config_text) is None:
        raise AssertionError("config/drone.ini should use integer num_layers = 3")
    if re.search(r"(?m)^total_timesteps\s*=\s*3000000\s*$", config_text) is None:
        raise AssertionError("config/drone.ini should default to a 3M timestep smoke run")
    if re.search(r"(?m)^domain_randomization\s*=\s*0\.0\s*$", config_text) is None:
        raise AssertionError("config/drone.ini should default domain_randomization = 0.0 for nominal V0.11")
    for key in [
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
    ]:
        if re.search(rf"(?m)^{key}\s*=\s*0\.0\s*$", config_text) is None:
            raise AssertionError(f"config/drone.ini should expose {key} = 0.0")
    if re.search(r"(?m)^action_scale\s*=\s*1\.0\s*$", config_text) is None:
        raise AssertionError("config/drone.ini should expose baseline action_scale = 1.0")
    if re.search(r"(?m)^action_mode\s*=\s*0\s*$", config_text) is None:
        raise AssertionError("config/drone.ini should keep legacy action_mode = 0 by default")
    for key, value in [
        ("normalized_thrust_min", "0.0"),
        ("normalized_thrust_max", "1.0"),
    ]:
        if re.search(rf"(?m)^{key}\s*=\s*{re.escape(value)}\s*$", config_text) is None:
            raise AssertionError(f"config/drone.ini should expose {key} = {value}")

    if "agent->params.action_scale = env->action_scale;" not in drone_h_text:
        raise AssertionError("env.action_scale is not wired into drone params")
    if "agent->params.action_mode = env->action_mode;" not in drone_h_text:
        raise AssertionError("env.action_mode is not wired into drone params")
    if "DomainRandomization dr = env_domain_randomization(env);" not in drone_h_text:
        raise AssertionError("env domain randomization config is not materialized on reset")
    if "init_drone(agent, &env->rng, &dr);" not in drone_h_text:
        raise AssertionError("granular domain randomization is not wired into init_drone")
    if "env->oob_radius = dict_get(kwargs, \"oob_radius\")->value;" not in binding_text:
        raise AssertionError("env.oob_radius is not wired through binding.c")
    if "env->action_mode = (int)dict_get_default(kwargs, \"action_mode\"" not in binding_text:
        raise AssertionError("env.action_mode is not wired through binding.c")
    if "env->normalized_thrust_max = dict_get_default(kwargs, \"normalized_thrust_max\"" not in binding_text:
        raise AssertionError("normalized thrust caps are not wired through binding.c")
    for key in [
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
    ]:
        if f'env->{key} = dict_get(kwargs, "{key}")->value;' not in binding_text:
            raise AssertionError(f"env.{key} is not wired through binding.c")
    if "> env->oob_radius" not in drone_h_text:
        raise AssertionError("OOB check is not using env.oob_radius")
    dronelib_text = DRONELIB.read_text(encoding="utf-8", errors="replace")
    if "actions[i] * params->action_scale" not in dronelib_text:
        raise AssertionError("actions are not scaled around hover trim")
    for expected in [
        "typedef struct {\n    float enabled;",
        "float com_x;",
        "drone->params.gravity = BASE_GRAVITY;",
        "BASE_MOTOR_FL_X - com_x",
        "hover_trim_thrusts(&drone->params, trim);",
        "#define MAX_ACTION_LATENCY_STEPS 8",
    ]:
        if expected not in dronelib_text:
            raise AssertionError("granular domain randomization support is incomplete")
    for expected in [
        "apply_action_latency(agent, raw_actions, env_action_latency_steps(env), delayed_actions);",
        "move_drone(agent, delayed_actions);",
        "env->sensor_noise > 0.0f",
    ]:
        if expected not in drone_h_text:
            raise AssertionError("latency/noise support is incomplete")

    com_x = 0.01
    com_y = -0.007
    shifted_motors = [(x - com_x, y - com_y) for x, y in motors]
    shifted_allocation = [
        [1.0, 1.0, 1.0, 1.0],
        [shifted_motors[i][1] for i in range(4)],
        [-shifted_motors[i][0] for i in range(4)],
        [k_drag * yaw[i] for i in range(4)],
    ]
    shifted_trim = solve_4x4(shifted_allocation, hover_target)
    shifted_achieved = mat_vec(shifted_allocation, shifted_trim)
    for i, (actual, expected) in enumerate(zip(shifted_achieved, hover_target)):
        close(actual, expected, 1e-9, f"COM-shifted hover allocation row {i}")
    if not all(v > 0.0 and math.isfinite(v) for v in shifted_trim):
        raise AssertionError(f"COM-shifted trim invalid: {shifted_trim}")

    print("Matrice 4D V0 checks passed")
    print("motor_order:", motor_order)
    print("hover_trim_N:", [round(v, 6) for v in trim])
    print("hover_rpm:", round(hover_rpm, 3))
    print("positive_yaw_torque:", round(yaw_torque, 6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
