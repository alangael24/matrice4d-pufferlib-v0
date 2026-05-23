# PPO-PAL v0.15 Pilot

This pilot keeps the v0.14 deployment contract intact so the v0.14 checkpoint
can be warm-started without surgery:

```text
action_mode = 0
action_scale = 0.7
OBS_SIZE = 23
policy.num_layers = 3
network = MinGRU
```

Implemented pieces:

```text
recurrent latent adaptation:
  existing MinGRU hidden state

probe-conditioned rollout:
  env.pal_probe_prob
  env.pal_probe_steps
  env.pal_probe_amp

Pareto-anchor guard:
  structured DR sampler via dr_profile_mix=3.0
  low LR from v0.14
  no EPOpt tail overweighting by default
  checkpoint selection by full suite, not latest

smooth/reset guard:
  env.alpha_action_delta
  env.alpha_reset_action_delta
  env.reset_action_interval
```

Main run:

```bash
BASE=/workspace/matrice4d-pufferlib-v0/baselines/general/policy.bin \
BATCH_ID=v0_15_ppo_pal_probe_anchor_from_v014_seed44_500m \
TOTAL_TIMESTEPS=500000000 \
SEED=44 \
GPU_ID=0 \
BUILD_FIRST=0 \
bash experiments/run_v0_15_ppo_pal_from_v014.sh
```

Ablation matrix:

```bash
BASE=/workspace/matrice4d-pufferlib-v0/baselines/general/policy.bin \
TOTAL_TIMESTEPS=100000000 \
SEED=44 \
GPU_ID=0 \
BUILD_FIRST=0 \
bash experiments/run_v0_15_ppo_pal_ablation_matrix.sh
```

The run script writes a minimal artifact at:

```text
artifacts/<BATCH_ID>_minimal.tgz
```

That archive contains the latest checkpoint, metadata, command, git status,
git diff, sha256, and compact JSON logs when present.

Not implemented in this pilot:

```text
multi-teacher KL inside the native CUDA PPO loss
privileged critic heads
auxiliary dynamics prediction heads
expanded actor observation tensor
```

Those require either a new native multi-policy training path or a slower
PyTorch backend pass, and they would not load v0.14 unchanged if the actor
observation shape changes.
