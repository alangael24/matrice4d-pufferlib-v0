# V0.11 Domain Randomization

Scope: add granular DR-light controls for robustness training without changing
nominal behavior when DR is disabled.

## Controls

All granular controls default to `0.0`. `domain_randomization` is the enable
switch for granular DR. With all `dr_*` values at zero, `domain_randomization`
also supports the legacy scalar range.

```bash
--env.domain-randomization 1
--env.dr-mass 0.05
--env.dr-inertia 0.10
--env.dr-k-thrust 0.10
--env.dr-linear-drag 0.20
--env.dr-yaw-drag 0.20
--env.dr-motor-lag 0.05
--env.dr-com-xy 0.01
--env.dr-com-z 0.00
--env.action-latency 0.00
--env.sensor-noise 0.00
```

## Sampling

Each reset samples per-agent physics parameters:

```text
mass        = BASE_MASS * U(1 - dr_mass, 1 + dr_mass)
Ixx/Iyy/Izz = BASE_I*   * U(1 - dr_inertia, 1 + dr_inertia)
k_thrust    = BASE_K_THRUST * U(1 - dr_k_thrust, 1 + dr_k_thrust)
b_drag      = BASE_B_DRAG   * U(1 - dr_linear_drag, 1 + dr_linear_drag)
k_drag      = BASE_K_DRAG   * U(1 - dr_yaw_drag, 1 + dr_yaw_drag)
k_mot       = BASE_K_MOT    * U(1 - dr_motor_lag, 1 + dr_motor_lag)
COM x/y     = U(-dr_com_xy, dr_com_xy)
COM z       = U(-dr_com_z, dr_com_z)
```

Ranges are clamped internally to keep multiplicative parameters positive.

`action_latency` is specified in seconds and rounded to the nearest 100 Hz
action step. It is capped by `MAX_ACTION_LATENCY_STEPS`.

`sensor_noise` is uniform noise added to each normalized observation element
after observation construction. It defaults to zero and is clamped internally.

## COM And Hover Trim

Motor lever arms are expressed relative to the sampled COM:

```text
x_i_eff = BASE_MOTOR_X_i - com_x
y_i_eff = BASE_MOTOR_Y_i - com_y
```

Hover trim is recomputed after sampling, so action `0` remains centered on the
current sampled mass, thrust coefficient, yaw drag, and COM lever arms.

## Logged Metrics

Episode logs include sample means:

```text
mass_mult_mean
ixx_mult_mean
iyy_mult_mean
izz_mult_mean
k_thrust_mult_mean
linear_drag_mult_mean
yaw_drag_mult_mean
motor_lag_mult_mean
com_x_mean
com_y_mean
com_z_mean
```

The current PufferLib log aggregator averages all float fields by episode
count. Global min/max are intentionally not exposed yet to avoid misleading
statistics.

## Validation

Run before GPU training:

```bash
python ocean/drone/matrice4d_v0_checks.py
bash build.sh drone --local
bash build.sh drone
```

Required properties:

```text
DR=0 keeps nominal parameters.
All sampled multiplicative parameters stay positive.
COM-offset allocation remains nonsingular.
Hover trim is finite and positive.
Action 0 remains approximately hover for each sampled agent.
```
