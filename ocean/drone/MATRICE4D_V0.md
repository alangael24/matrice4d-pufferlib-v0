# Matrice 4D V0

This branch turns `ocean/drone` into a Matrice 4D CAD-aligned flight-dynamics
V0 for hover and go-to-point RL.

It is not a full physical digital twin. Gimbal, camera, thermal, exact COM,
measured inertia, measured motor curves, measured drag, and confirmed DJI rotor
spin directions are out of scope for V0.

## Scope

- 6-DoF flight dynamics using Matrice-scale mass and estimated inertia.
- CAD motor order: `[FL, FR, RL, RR]`.
- CAD motor datums in meters:
  - `FL = (-0.1915, +0.1708, 0.0)`
  - `FR = (+0.1915, +0.1708, 0.0)`
  - `RL = (-0.1715, -0.1708, 0.0)`
  - `RR = (+0.1715, -0.1708, 0.0)`
- Allocation matrix:
  - `T_total = sum(T_i)`
  - `tau_x = sum(y_i * T_i)`
  - `tau_y = sum(-x_i * T_i)`
  - `tau_z = k_drag * sum(yaw_sign_i * T_i)`
- Action `0` maps to hover trim from the CAD allocation matrix.
- `env.action_scale` scales policy actions around hover trim. `1.0` preserves
  the full baseline range; `0.2` or `0.3` is intended for easy curriculum runs.
- `env.domain_randomization` controls the per-reset physics randomization
  amount. `0.05` is the baseline; `0.0` disables it.
- Observations remain the PufferLib 23-float drone observation vector, with
  body-frame velocity and body-frame target vector.
- Render fallback uses the same CAD motor positions instead of the original
  symmetric Crazyflie visual layout.

## Known Assumptions

- `yaw_sign = [+1, -1, -1, +1]` is an assumed diagonal pairing, not confirmed
  DJI hardware data.
- COM projection is assumed to coincide with the CAD motor-plane origin.
- `Ixx/Iyy/Izz`, `k_thrust`, `k_drag`, drag, and motor lag are first-pass
  estimates intended for simulator training and debugging.
- This V0 must not be used for real motor-level deployment.

## Quick Check

Run from the PufferLib repo root:

```bash
python ocean/drone/matrice4d_v0_checks.py
```

Expected output includes:

```text
Matrice 4D V0 checks passed
hover_rpm: 5525.0
```

## First Smoke Training

`config/drone.ini` is set for a short first run:

```text
num_layers = 3
total_timesteps = 3000000
domain_randomization = 0.05
action_scale = 1.0
```

Use the short run to catch NaNs, reset bugs, unstable rewards, and visualization
issues before launching a longer 40M+ timestep training job.
