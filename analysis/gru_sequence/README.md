# GRU Sequence: Bidirectional GRU vs. Bidirectional Temporal CNN

Two experiments against the existing `analysis/spatial_sequence` work, given the identical
24-step, 32-channel Cyranose window and the identical leave-one-session-out
area-classification protocol:

- **v1**: a strict single-variable isolation -- temporal convolution vs. recurrent memory,
  with everything else (capacity, dropout, normalization, learning rate, data, labels,
  splits, seeds) held fixed to the existing TCN/Transformer's own settings.
- **v2**: v1 underperformed substantially (see `analysis/reports/gru-sequence/report.md`),
  so v2 asks a different, explicitly-broader question -- can a GRU *reasonably tuned for
  itself* be made competitive? Capacity, normalization, dropout, and learning rate all
  change together. This trades clean single-variable attribution for a practical answer;
  both experiments are kept and reported side by side rather than v2 replacing v1.

This experiment uses **only the 5 already-annotated July 30 sessions**, unchanged. No new
annotation work was needed, so unlike `spatial_sequence` this package has no `annotation.py`.

## What's reused vs. new

Everything data/label/loss/metric-related is imported unchanged from `analysis.spatial_sequence`:
`build_prepared_data`, `SpatialWindowDataset`/`SpatialAnchorDataset`, `train_one_epoch`,
`validation_loss`, `evaluate_and_save`, `checkpoint_selection_metric`, `selection_is_better`,
`parameter_groups`, and the checkpoint/seeding utilities. No existing `spatial_sequence` file
was modified.

New in this package:

- `models.py` -- `BiGRUAreaModel` (v1) and `BiGRUAreaModelV2` (v2), both sharing the same
  `input_projection = nn.Linear(32, 96)` as the existing TCN/Transformer (so "raw channels ->
  representation" is identical across all architectures compared) and the same dense,
  per-timestep output contract.
  - **v1**: hidden_size=64/direction (128-dim concat), no dropout, no normalization.
  - **v2**: hidden_size=161/direction (322-dim concat, ~matches the TCN's parameter count),
    LayerNorm on the GRU's output (the single-block analogue of the TCN's per-block
    pre-norm), Dropout(0.1) (matching the existing models' dropout, previously inert here
    since `nn.GRU`'s own `dropout=` argument only activates between stacked layers and we
    use `num_layers=1`). No residual connection -- deferred; a plain residual isn't
    shape-compatible without a learned projection (96-dim input vs. 322-dim GRU output), and
    a single recurrent block doesn't have the same across-many-blocks gradient problem a
    residual mainly helps with. Would be the next addition if v2 still underperforms.
  - v1 is left untouched by v2's existence -- it's a separate class, not a parameter change,
    so the original single-variable results stay exactly reproducible.
- `training.py` -- a trimmed `RunConfig` + `train_run`/`evaluate_checkpoint`, restricted to
  this experiment's fixed configuration (target=area, temporal_mode=bidirectional,
  evaluation_scheme=leave-one-session-out), with an `architecture_version` field selecting
  v1 vs v2. Deliberately does not support checkpoint resume or periodic epoch snapshots
  (runs are ~1-2 minutes each; not needed at this scale) -- everything else (epoch loop
  mechanics, checkpoint-selection criterion, seeding) mirrors
  `spatial_sequence.training.train_run` as closely as possible given the different config type.
- `report.py` -- aggregates the runs (both versions, if present) into the comparison report,
  including a 5-seed majority-vote confusion matrix per held-out session per version (ties
  broken by higher mean probability then lower class index, matching the convention
  documented for the existing TCN figures).

## Parameter counts and learning rates

| Model | Parameters | Learning rate |
|---|---:|---|
| Existing bidirectional Temporal CNN (area head) | 254,787 | 1e-4 |
| Existing bidirectional Transformer (area head) | 228,003 | 1e-4 |
| BiGRU v1 (hidden=64) | 65,763 | 1e-4 (unchanged from TCN/Transformer, deliberately, to isolate one variable) |
| BiGRU v2 (hidden=161, +dropout, +LayerNorm) | 254,975 | 1e-3 (chosen via a 4-candidate probe: 1e-4/3e-4/1e-3/3e-3, 30 epochs, one fold, one seed) |

v2's parameter count is within 0.07% of the TCN's (254,975 vs 254,787).

### Why 1e-3 for v2

Holding the TCN's learning rate fixed for a structurally different, differently-sized model
isn't obviously right, so a short probe (not a full sweep) picked it: on
`mint-only-horizontal-run01` (seed 0, 30 epochs), macro-F1 was 0.3726 / 0.3696 / **0.4044** /
0.3987 for lr = 1e-4 / 3e-4 / 1e-3 / 3e-3 respectively. 1e-3 won on both metric value and
qualitative behavior (zero false-lavender predictions on a session with no lavender present,
unlike 3e-3). Tuned on that one fold only, then held fixed across all 5 folds for the real
sweep, so the other 4 folds' reported numbers aren't touched by the tuning process.

## Comparison scope

The existing leave-one-session-out study (`analysis/reports/spatial-sequence/leave-one-session-out/report.md`)
was run only for the Temporal CNN, and only reports macro-F1 mean +/- SD per session -- no
confusion-matrix numbers, no per-class precision/recall, no balanced accuracy, and no
Transformer LOSO run exists at all (Transformer's LOSO cell is simply absent; its
within-session numbers use a different protocol and aren't comparable). Per an explicit
scope decision, this experiment does **not** retrain the TCN or Transformer to fill that gap --
the report quotes the existing TCN macro-F1 numbers as-is, and reports the GRU's fuller
metric set (both versions) on its own.

## Commands

Train one fold (v1 or v2):

```bash
python -m analysis.gru_sequence --held-out-session mint-only-horizontal-run01 --seed 0 --epochs 100
python -m analysis.gru_sequence --held-out-session mint-only-horizontal-run01 --seed 0 --epochs 100 \
  --architecture-version v2 --learning-rate 1e-3
```

Run a full sweep (5 held-out sessions x 5 seeds = 25 runs):

```bash
bash output/gru-sequence/run_sweep.sh      # v1, ~25-30 minutes on CPU
bash output/gru-sequence/run_sweep_v2.sh   # v2, ~60-70 minutes on CPU (larger model)
```

Re-run the learning-rate probe:

```bash
bash output/gru-sequence/lr_probe.sh
```

Generate the comparison report from completed runs (works with v1 only, or v1+v2):

```bash
python -m analysis.gru_sequence --aggregate-report
```

Evaluate a specific saved checkpoint:

```bash
python -m analysis.gru_sequence --eval-only --checkpoint <path to best.pt>
```

Generated artifacts:

```text
analysis/reports/gru-sequence/
  report.md
  summary.json

output/gru-sequence/
  v1/leave-one-session-out/runs/area/<held_out_session>/bigru_bidirectional_seed<seed>/
  v2/leave-one-session-out/runs/area/<held_out_session>/bigru_bidirectional_seed<seed>/
    best.pt, history.csv, metrics.json, aggregated_predictions.npz,
    resolved_config.json, data_signature.json
  lr_probe/lr_<rate>/          -- the 4 probe runs (30 epochs each, seed 0 only)
  logs/
  run_sweep.sh, run_sweep_v2.sh, lr_probe.sh
```
