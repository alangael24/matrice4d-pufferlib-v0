# Drone Metrics

These metrics are instrumentation only. They do not change reward, dynamics, action
mapping, observations, reset distribution, or OOB logic.

PufferLib reports these values over completed episodes. Per-step quantities are
averaged inside each episode first, then averaged across episodes by the vecenv
log aggregator.

## Existing

- `perf`: Final hover EMA averaged over completed episodes.
- `score`: Sum of hover checks over the episode.
- `oob`: Fraction of completed episodes that ended out of bounds.
- `timeout`: Fraction of completed episodes that reached the horizon.
- `episode_return`: Episode reward sum.
- `episode_length`: Episode length in environment steps.
- `ema_dist`: Final EMA of target distance.
- `ema_vel`: Final EMA of linear velocity norm.
- `ema_omega`: Final EMA of angular velocity norm.

## Angular Velocity

- `ema_omega_x`: Final EMA of `abs(omega.x)`, body roll-rate component.
- `ema_omega_y`: Final EMA of `abs(omega.y)`, body pitch-rate component.
- `ema_omega_z`: Final EMA of `abs(omega.z)`, body yaw-rate component.

Use these to separate yaw spin from roll/pitch oscillation.

## Actions And Saturation

- `mean_abs_action`: Legacy alias for `mean_abs_action_raw`.
- `max_abs_action`: Legacy alias for `max_abs_action_raw`.
- `action_saturation_frac`: Legacy diagnostic alias for raw policy action values
  with `abs(action) >= 0.99`. This is not necessarily env clipping.
- `mean_abs_action_raw`: Mean absolute raw policy action before env clamp.
- `max_abs_action_raw`: Maximum absolute raw policy action before env clamp.
- `raw_action_clip_frac`: Fraction of raw policy actions outside `[-1, 1]`.
  This is true env action clipping.
- `mean_abs_action_clipped`: Mean absolute action after the env clamp.
- `max_abs_action_clipped`: Maximum absolute action after the env clamp.
- `clipped_action_saturation_frac`: Fraction of clipped actions with
  `abs(clipped_action) >= 0.99`.
- `motor_clip_low_frac`: Fraction of motor thrust targets at or below zero after
  `target_thrust = hover_trim * (1 + action_scale * clipped_action)`.
- `motor_clip_high_frac`: Fraction of motor thrust targets at or above the
  configured max motor thrust after the same hover-trim mapping.

Use `raw_action_clip_frac` to detect real env clipping. Use
`action_saturation_frac` and `clipped_action_saturation_frac` to detect whether
the policy is pushing against the action boundary. `motor_clip_high_frac` can
remain zero when `action_scale` is small because the hover-trim mapping still
keeps motor thrust targets inside the physical range.

## Motors

- `mean_rpm_FL`: Mean actual RPM for front-left motor.
- `mean_rpm_FR`: Mean actual RPM for front-right motor.
- `mean_rpm_RL`: Mean actual RPM for rear-left motor.
- `mean_rpm_RR`: Mean actual RPM for rear-right motor.

Motor order is `[FL, FR, RL, RR]`, matching the CAD datums and action order.

## Reward Components

- `r_dist`: Episode sum of the distance-progress reward component.
- `r_hover`: Episode sum of the hover-potential reward component.
- `r_shaping`: Episode sum of the potential-difference shaping component.
- `r_omega`: Episode sum of the angular-velocity penalty component.
- `r_terminal`: Episode sum of terminal reward/penalty. This is currently `0.0`
  because V0 does not add a terminal reward term.

These components should sum to approximately `episode_return`, subject to
floating-point accumulation.
