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

- `mean_abs_action`: Mean absolute raw policy action before the environment clamp.
- `max_abs_action`: Maximum absolute raw policy action seen in the episode.
- `action_saturation_frac`: Fraction of raw policy action values with
  `abs(action) >= 0.99`.
- `motor_clip_low_frac`: Fraction of motor thrust targets at or below zero after
  `target_thrust = hover_trim * (1 + action_scale * clipped_action)`.
- `motor_clip_high_frac`: Fraction of motor thrust targets at or above the
  configured max motor thrust after the same hover-trim mapping.

`action_saturation_frac` can be high while `motor_clip_high_frac` is zero when
`action_scale` is small. That means the policy is pushing against the action
interface, but the motor target is still limited by the curriculum scale.

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
