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
python ocean/drone/cuda_checks.py
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
CUDA_CHECK_STEPS=200
PROFILE_TOTAL_AGENTS=32768
PROFILE_BUFFERS=8
PROFILE_THREADS=32
PROFILE_HORIZON=256
```

`CUDA_CHECK_STEPS=200` is the strict smoke gate. Longer open-loop checks can be run manually, for example:

```bash
CUDA_CHECK_STEPS=1000 RUN_TRAIN=0 bash tools/benchmark_cuda_env.sh
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
correctness: matrice4d_v0_checks and cuda_checks pass
speed: CUDA envspeed throughput > CPU envspeed throughput
quality: CUDA training reaches similar env metrics in less wall-clock time
```
