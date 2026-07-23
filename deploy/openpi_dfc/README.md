# π0.5 base for Dual Flexiv

This overlay serves the public `pi05_base` weights with the DFC contract:

- state: 14 joints (`left q[7], right q[7]`);
- action: 16 values (`left q[7], grip, right q[7], grip`);
- joint actions are normalized as deltas from the current joint state;
- gripper actions stay absolute;
- `static_left` fills π0.5's base-camera slot;
- both wrist-camera slots are black and masked.

It intentionally does not use `serve_policy.py --env ALOHA`. That shortcut
selects π0.5 base parameters but applies ALOHA's 6-DoF-per-arm joint/gripper
adapter and expects top/wrist camera keys.

The checked-in normalization stats are generated from
`datasets/dfc/handover_bimanual_v2`, the current 14-state/16-action dataset with
0–1 grippers:

```bash
python deploy/openpi_dfc/compute_norm_stats.py \
  --dataset datasets/dfc/handover_bimanual_v2 \
  --output deploy/openpi_dfc/norm_stats.json
```

Run the overlay from an OpenPI checkout/container:

```bash
uv run deploy/openpi_dfc/serve_dfc_policy.py \
  --checkpoint gs://openpi-assets/checkpoints/pi05_base \
  --norm-stats deploy/openpi_dfc/norm_stats.json \
  --port 8000
```

The base weights are not fine-tuned on DFC demonstrations. Keep follower
control inhibited and validate the returned trajectories as a dry run before
considering an explicitly fine-tuned checkpoint.
