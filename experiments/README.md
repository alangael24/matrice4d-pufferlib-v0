# Matrice 4D Experiment Runners

This directory contains reproducible GPU batch runners. These scripts do not
change environment physics, rewards, observations, or reset behavior. They only
standardize how runs are launched and how artifacts are preserved.

## Quick Start On A Rental GPU

```bash
git clone -b feat/v0.1-artifacts-runner https://github.com/alangael24/matrice4d-pufferlib-v0.git
cd matrice4d-pufferlib-v0
pip install -e . --no-build-isolation
bash build.sh drone
bash experiments/run_v0_1_gpu_batch.sh
```

The batch writes into:

```text
runs/<batch_id>/<run_name>/
```

Each run directory contains:

```text
checkpoints/
logs/
stdout.txt
command.sh
summary.json
checkpoint_summary.txt
git_commit.txt
git_diff.patch
```

At the end, the runner creates:

```text
artifacts/<batch_id>.tgz
artifacts/<batch_id>.tgz.sha256
```

Download that `.tgz` from the GPU before destroying the instance.

## Useful Overrides

For a smoke pass:

```bash
TOTAL_TIMESTEPS_PHASEA=1000000 \
TOTAL_TIMESTEPS_TARGET2=1000000 \
TOTAL_TIMESTEPS_TARGET5=1000000 \
bash experiments/run_v0_1_gpu_batch.sh
```

To select a GPU:

```bash
GPU_ID=0 bash experiments/run_v0_1_gpu_batch.sh
```

To add extra Puffer CLI args to every run:

```bash
EXTRA_ARGS="--train.minibatch-size 8192" bash experiments/run_v0_1_gpu_batch.sh
```

If this branch is later merged with OOB config, target-5m runs can use:

```bash
EXTRA_ARGS="--env.oob-radius 12" bash experiments/run_v0_1_gpu_batch.sh
```

## V0.11 DR Ablation Sweep

Use this runner to find which DR-medium parameter group breaks target5. It
continues from the V0.10 target5 nominal checkpoint and starts from the known
passing DR-light configuration. Each ablation raises one parameter group to the
DR-medium setting.

Minimal first pass:

```bash
BASE=/matrice4d-pufferlib-v0/base_target5_seed46.bin \
BATCH_ID=v0_11_dr_ablation_core_100m \
TOTAL_TIMESTEPS=100000000 \
bash experiments/run_v0_11_dr_ablation.sh
```

The default ablations are:

```text
k_thrust motor_lag latency_noise
```

Full sweep:

```bash
BASE=/matrice4d-pufferlib-v0/base_target5_seed46.bin \
BATCH_ID=v0_11_dr_ablation_full_100m \
TOTAL_TIMESTEPS=100000000 \
ABLATIONS="all" \
bash experiments/run_v0_11_dr_ablation.sh
```

Explicit custom sweep:

```bash
BASE=/matrice4d-pufferlib-v0/base_target5_seed46.bin \
ABLATIONS="com inertia drag mass medium_all" \
bash experiments/run_v0_11_dr_ablation.sh
```

Valid ablations:

```text
light_control
mass
inertia
k_thrust
drag
motor_lag
com
latency_noise
latency
sensor_noise
medium_all
```

## V0.11 K-Thrust Authority Sweep

Use this runner after DR-light passes but DR-medium fails. It tests whether
`k_thrust` randomization failure is caused by insufficient control authority.
All non-thrust DR stays at the DR-light setting.

Default sweep:

```text
kt15_scale05
kt20_scale05
kt20_scale07
kt20_scale08
```

Run:

```bash
BASE=/matrice4d-pufferlib-v0/base_dr_light_seed46.bin \
BATCH_ID=v0_11_kthrust_authority_100m \
TOTAL_TIMESTEPS=100000000 \
bash experiments/run_v0_11_kthrust_authority_sweep.sh
```

Short smoke test:

```bash
BASE=/matrice4d-pufferlib-v0/base_dr_light_seed46.bin \
SEEDS="42 46" \
TOTAL_TIMESTEPS=50000000 \
bash experiments/run_v0_11_kthrust_authority_sweep.sh
```

Important metrics:

```text
hover_trim_rpm_mean
hover_trim_rpm_max
hover_trim_rpm_frac_of_max
motor_clip_high_frac
motor_clip_low_frac
mean_abs_action_clipped
action_saturation_frac
k_thrust_mult_min
k_thrust_mult_max
```
