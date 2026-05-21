# Sim-to-Real Normalized Thrust Mode

This is an opt-in action interface for training new motor-level policies.
It does not change the existing V0 policy interface by default.

## Modes

`action_mode = 0` keeps the current Matrice 4D V0 behavior:

```text
action = clamp(policy_action * action_scale, -1, 1)
action 0 -> hover trim thrust from CAD allocation
action +1 -> max motor thrust
action -1 -> zero thrust
```

`action_mode = 1` uses normalized motor thrust:

```text
f_hat = 0.5 * (clamp(policy_action, -1, 1) + 1)
f_hat = clamp(f_hat, normalized_thrust_min, normalized_thrust_max)
target_thrust = f_hat * max_motor_thrust
```

The motor/action order is unchanged:

```text
[FL, FR, RL, RR]
```

## Intended Use

Use `action_mode = 1` for new sim-to-real policies that should learn a common
motor command representation across a bounded family of quadrotors.

Keep `action_mode = 0` when evaluating existing checkpoints such as the current
DR-medium robust baseline. Those policies were trained around hover trim and
are not expected to behave correctly under normalized thrust without retraining.

## Safety Caps

For high-thrust hardware, train and evaluate with conservative caps first:

```ini
action_mode = 1
normalized_thrust_min = 0.0
normalized_thrust_max = 0.5
```

The cap is a simulator/runtime command limit. Hardware deployment still needs
separate ESC/PX4 output limits, watchdogs, kill switch, motor order checks, and
thrust-stand calibration.
