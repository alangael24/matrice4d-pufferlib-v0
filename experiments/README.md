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
