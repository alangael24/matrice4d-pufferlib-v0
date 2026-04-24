# Experiment Protocol

This protocol is for rental GPU runs where losing checkpoints/logs is the main
operational risk.

## Setup

```bash
git clone -b feat/v0.1-artifacts-runner https://github.com/alangael24/matrice4d-pufferlib-v0.git
cd matrice4d-pufferlib-v0
pip install -U pip setuptools wheel
pip install -e . --no-build-isolation
bash build.sh drone
```

If the machine is missing the NVML symlink:

```bash
ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so
ldconfig
bash build.sh drone
```

## Run The Batch

```bash
GPU_ID=0 bash experiments/run_v0_1_gpu_batch.sh
```

The batch includes:

1. Checkpoint smoke test.
2. Phase A close-hover seed46 30M.
3. Target 2 m seeds 42-46.
4. Target 5 m action-scale sweep: 0.2, 0.3, 0.4, 0.5.

## Artifact Contract

Each important run must preserve:

- `stdout.txt`
- `command.sh`
- `run_metadata.env`
- `checkpoints/`
- `logs/`
- `summary.json`
- `checkpoint_summary.txt`
- `git_commit.txt`
- `git_status.txt`
- `git_diff.patch`

The batch root also preserves:

- `build_stdout.txt`
- `summary.json`
- `summary.csv`
- `artifact_manifest.txt`

The final archive is:

```text
artifacts/<batch_id>.tgz
artifacts/<batch_id>.tgz.sha256
```

## Download Before Destroying The GPU

From Windows PowerShell:

```powershell
scp -P <PORT> -i "$HOME\private_key.pem" `
  root@<HOST>:/root/matrice4d-pufferlib-v0/artifacts/<batch_id>.tgz `
  "$HOME\Downloads\<batch_id>.tgz"
```

Download the `.sha256` too when available:

```powershell
scp -P <PORT> -i "$HOME\private_key.pem" `
  root@<HOST>:/root/matrice4d-pufferlib-v0/artifacts/<batch_id>.tgz.sha256 `
  "$HOME\Downloads\<batch_id>.tgz.sha256"
```

Do not rely on Git for model weights. Checkpoints are ignored by `.gitignore`
and must be copied as artifacts.

## Inspect On GPU

List checkpoints:

```bash
bash scripts/find_checkpoints.sh runs/<batch_id>
```

Summarize all completed run stdout files:

```bash
python scripts/summarize_runs.py \
  --runs-root runs/<batch_id> \
  --output runs/<batch_id>/summary.json \
  --csv runs/<batch_id>/summary.csv
```

`summary.csv` and `summary.json` are generated from `stdout.txt` by taking the
last occurrence of each PufferLib metric. This avoids stale early-frame metrics
from interactive terminal redraws.
