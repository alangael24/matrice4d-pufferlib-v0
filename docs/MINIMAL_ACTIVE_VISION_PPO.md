# Minimal Active Vision PPO

This branch tests the thesis that PPO can learn from very low-bandwidth pixels
when the pixels are a causal control sensor instead of a general image.

## Observation Contract

The drone observation is now 32 floats:

- 23 legacy proprio/state floats
- 9 minimal vision floats: `3 pixels x RGB`

The main pilot uses:

```text
minimal_vision_enabled = 1
minimal_vision_mask_target = 1
minimal_vision_only = 0
```

That means the policy keeps body velocity, angular velocity, attitude, and RPMs,
but the perfect target vector and target normal are zeroed. Direction/proximity to
the target must come from the 1x3 RGB retina.

Strict 9-float-only mode is available:

```text
MINIMAL_VISION_ONLY=1 MINIMAL_VISION_MASK_TARGET=0
```

That is intentionally harder because the current action interface is direct
4-motor low-level control, not CTBR/body-rate control.

## Sensor

The 1x3 retina is not a renderer. It is a structured active sensor:

```text
pixel 0: left angular sector
pixel 1: center angular sector
pixel 2: right angular sector
channels: RGB intensity from target bearing, vertical alignment, and distance
```

The default curriculum keeps the target initially visible:

```text
minimal_vision_spawn_visible_target = 1
```

so the first run tests visual servoing before testing recovery/search.

## Run

```bash
BUILD_FIRST=1 \
BATCH_ID=v0_minimal_active_vision_3x1_targetmask_seed44_100m \
TOTAL_TIMESTEPS=100000000 \
SEED=44 \
GPU_ID=0 \
bash experiments/run_minimal_active_vision_ppo.sh
```

Useful harder variants:

```bash
# strict 9-float retina only
MINIMAL_VISION_ONLY=1 MINIMAL_VISION_MASK_TARGET=0 \
BATCH_ID=v0_minimal_active_vision_3x1_strict_seed44_100m \
bash experiments/run_minimal_active_vision_ppo.sh

# add visual distractors
MINIMAL_VISION_DISTRACTORS=0.25 \
BATCH_ID=v0_minimal_active_vision_3x1_distractors_seed44_100m \
bash experiments/run_minimal_active_vision_ppo.sh

# remove visible-spawn curriculum
MINIMAL_VISION_SPAWN_VISIBLE_TARGET=0 \
BATCH_ID=v0_minimal_active_vision_3x1_search_seed44_100m \
bash experiments/run_minimal_active_vision_ppo.sh
```

## What Counts As Progress

This is not a deployment policy. The first pass should be judged by:

- lower OOB than the previous broken camera run
- nonzero target-seeking behavior from masked target observations
- stable velocity/omega with direct motor actions
- sensitivity to the 1x3 RGB target signal

If this works, the next clean step is a CTBR/body-rate action interface or a
teacher/student version for gate racing.
