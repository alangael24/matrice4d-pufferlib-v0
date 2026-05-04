# CUDA Env Benchmark

Benchmark completo para validar el entorno CUDA de Matrice 4D contra el entorno CPU usado por PufferLib.

## Smoke Correctness And Speed

```bash
RUN_TRAIN=0 bash tools/benchmark_cuda_env.sh
```

Esto corre:

```text
python ocean/drone/matrice4d_v0_checks.py
bash build.sh drone
python ocean/drone/cuda_checks.py --steps 1000 --num-agents 128 --action-amplitude 0.0 --obs-tol 1e-6 --reward-tol 1e-5
python ocean/drone/cuda_checks.py --steps 200 --num-agents 128 --action-amplitude 0.05
python ocean/drone/cuda_checks.py --steps 1000 --num-agents 128 --action-amplitude 0.05 --dump-diffs ...
PUFFERLIB_DISABLE_ENV_CUDA=1 bash build.sh drone --profile
./profile envspeed ...
bash build.sh drone --profile
./profile envspeed ...
```

## End To End

```bash
bash tools/benchmark_cuda_env.sh
```

Defaults:

```text
SEEDS=42
TOTAL_TIMESTEPS=3000000
CUDA_ZERO_STEPS=1000
CUDA_ZERO_AGENTS=128
CUDA_ZERO_OBS_TOL=1e-6
CUDA_ZERO_REWARD_TOL=1e-5
CUDA_SMOKE_STEPS=200
CUDA_SMOKE_AGENTS=128
CUDA_SMOKE_ACTION_AMPLITUDE=0.05
CUDA_DIAG_STEPS=1000
CUDA_DIAG_AGENTS=128
CUDA_DIAG_ACTION_AMPLITUDE=0.05
PROFILE_TOTAL_AGENTS=32768
PROFILE_BUFFERS=8
PROFILE_THREADS=32
PROFILE_HORIZON=256
```

Required correctness gates:

```bash
python ocean/drone/cuda_checks.py \
  --steps 1000 \
  --num-agents 128 \
  --action-amplitude 0.0 \
  --obs-tol 1e-6 \
  --reward-tol 1e-5

python ocean/drone/cuda_checks.py \
  --steps 200 \
  --num-agents 128 \
  --action-amplitude 0.05
```

The 1000-step `action_amplitude=0.05` check is recorded as a non-blocking diagnostic because terminal-boundary divergence can reset CPU/CUDA trajectories differently after the first mismatch:

```bash
python ocean/drone/cuda_checks.py \
  --steps 1000 \
  --num-agents 128 \
  --action-amplitude 0.05 \
  --dump-diffs runs/<BENCH_ID>/correctness/diagnostics/cuda_amp005_1000.csv \
  --dump-state
```

Para cinco seeds:

```bash
SEEDS="1 2 3 4 5" bash tools/benchmark_cuda_env.sh
```

Artifacts:

```text
runs/<BENCH_ID>/summary.md
runs/<BENCH_ID>/summary.json
runs/<BENCH_ID>/train_summary.csv
artifacts/<BENCH_ID>.tgz
```

Success gates:

```text
correctness: matrice4d_v0_checks, zero-action 1000-step parity, and amp=0.05 200-step parity pass
speed: CUDA envspeed throughput > CPU envspeed throughput
quality: CUDA training reaches similar env metrics in less wall-clock time
```
