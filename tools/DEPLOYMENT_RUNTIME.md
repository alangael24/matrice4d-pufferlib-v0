# Matrice 4D Deployment Runtime Contract

This is the parity contract for loading and executing the native MinGRU drone
policy outside the trainer.

## Policy Runtime

- Observation size: `23`
- Action size: `4`
- Hidden size: `128`
- MinGRU layers: `3`
- Decoder outputs: `5` (`4` action means + `1` value head)
- Recurrence reset: every `32` environment steps by default
- Observation normalization: none
- Action output: raw policy mean actions
- Sim action handling for current V0 checkpoints: `action_mode = 0`, clamp raw
  actions to `[-1, 1]`, then apply `env.action_scale` around hover trim.
- Experimental sim-to-real policies can use `action_mode = 1`, which maps raw
  actions to normalized motor thrust with `normalized_thrust_min/max` caps. The
  current V0 baseline checkpoints were not trained for that mode.

## Checkpoint Layout

The native backend allocates tensors on 16-byte boundaries but saves
`master_weights.shape == total_elems`. That means the saved file has the flat
element count, while the in-memory tensor pointers use 8-float alignment.

The deployment runtime intentionally mirrors the C/Puffer evaluator:

```text
encoder weight                 128 * 23
align to 8 floats
decoder weight                 5 * 128
align to 8 floats
logstd                         4
align to 8 floats
mingru layer 0                 3 * 128 * 128
align to 8 floats
mingru layer 1                 3 * 128 * 128
align to 8 floats
mingru layer 2                 3 * 128 * 128
align to 8 floats
```

The file has `151044` floats (`604176` bytes). The runtime pads the loaded file
with 7 zero floats before assigning aligned pointers, matching the old
`get_weights_aligned` behavior.

## Gate

Run:

```bash
python3 tools/check_deployment_runtime_parity.py \
  --checkpoint /Users/alan/matrice4d_experiments/release_assets/v0.14-dr-medium-robust-baseline/v0_14_dr_medium_robust_seed44_scale0.7_reset32_latest.bin
```

The checker compares Python policy math against the C deployment runtime on the
same deterministic observation fixture. It reports raw action and scaled action
max absolute differences.
