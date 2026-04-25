# Drone OOB Radius And Targets

The hover/go-to-point task samples a target within `hover_target_dist` of the
drone reset position. Out-of-bounds termination is now controlled by
`env.oob_radius`, measured as distance from the current target:

```text
oob = distance(drone.position, target.position) > oob_radius
```

This branch only changes OOB configurability. It does not change reward, action
mapping, domain randomization, observations, or reset distribution.

## CLI

Use hyphenated CLI names for PufferLib overrides:

```bash
puffer train drone \
  --env.hover-target-dist 5 \
  --env.oob-radius 12
```

## Recommended Values

Use a larger OOB radius when the task target is farther away. The agent needs
room for acceleration, overshoot, and recovery while it is still learning.

```text
hover_target_dist 0.5-2 m: oob_radius 6-8 m
hover_target_dist 5 m:     oob_radius 12 m
```

The default `config/drone.ini` uses:

```text
hover_target_dist = 5.0
oob_radius = 12.0
```

## Interpretation

If `oob` remains high with `oob_radius = 12` on a 5 m target task, the failure is
less likely to be a too-small arena and more likely to be control instability,
action authority, reward balance, or reset difficulty.
