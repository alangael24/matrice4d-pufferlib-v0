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
