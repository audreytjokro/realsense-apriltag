# Spatial Sequence Pipeline

This package implements the annotation, training, anchor validation, position-video,
and reporting workflow for the five usable sessions in the 2026-07-30 long
random-trajectory pilot. All reported values are exploratory validation results;
there is no independent test set.

The model input is a sequence of the 32 raw flag-2 `pcnose_S*_kohm` channels.
Canonical and leave-one-session-out runs use length 24; the sequence-length
study additionally uses lengths 6, 12, and 18. Pose, timestamps, layout, session
identity, thermistor values, and annotation metadata are label/evaluation
information only.

Training uses a centered Train-Guard-Validation-Guard-Train split with two 23-row
guards. Training supervises all 24 tokens. Formal validation scores one shared
physical anchor row per window: causal index 23 and bidirectional index 12.
Position uses a paper-only 13x13 joint classification head and masks paper-exterior
position labels.

Input and regression-target statistics use every physical training row exactly
once. Validation rows never contribute to normalization.

## Annotation

Open `analysis/notebooks/long_random_sequence_spatial_annotation.ipynb` with the
`pap2` kernel. The five completed `spatial_annotation.json` files live beside their
session CSV and video. Notebook 09 is neither modified nor imported.

## Training and evaluation

Session selectors accept either the readable experiment slug or the timestamp.
New paths and reports use the slug.

```bash
CUDA_VISIBLE_DEVICES=0 python -m analysis.spatial_sequence \
  --architecture temporal-cnn \
  --temporal-mode bidirectional \
  --target position \
  --session lavender-only-horizontal-run01 \
  --seed 0 \
  --epochs 100 \
  --device cuda:0
```

Pooled targets omit `--session`:

```bash
python -m analysis.spatial_sequence \
  --architecture transformer \
  --temporal-mode causal \
  --target distance
```

Resume restores model, optimizer, scheduler, epoch, history, and Python/NumPy/
PyTorch/data-loader RNG state. Evaluation reloads the complete configuration:

```bash
python -m analysis.spatial_sequence --eval-only \
  --checkpoint output/spatial-sequence/runs/distance/transformer_causal_seed0/best.pt
```

`run_manifest.csv` enumerates 180 runs: 36 logical configurations across seeds
0-4. The unfinished rows can be dynamically assigned one process per GPU:

```bash
python -m analysis.spatial_sequence.scheduler --gpus 0,1,2,3,4,5,6,7,8,9
```

The sequence-length manifest contains 40 bidirectional Temporal CNN area/distance
rows: 30 new runs at lengths 6/12/18 and ten references to canonical length-24
runs. The LOSO manifest contains 50 bidirectional Temporal CNN area/distance
runs. In each LOSO fold, four complete sessions train the model and the fifth
complete session selects and reports the checkpoint; this is cross-session
validation, not an independent test.

```bash
python -m analysis.spatial_sequence.scheduler --gpus 0,1,2,3,4,5,6,7,8,9 \
  --manifest analysis/spatial_sequence/sequence_length_manifest.csv

python -m analysis.spatial_sequence.scheduler --gpus 0,1,2,3,4,5,6,7,8,9 \
  --manifest analysis/spatial_sequence/leave_one_session_out_manifest.csv
```

Generate either study report by supplying the corresponding manifest to
`--aggregate-report`.

Generate the English report, 64 summary figures, and 20 representative videos:

```bash
python -m analysis.spatial_sequence --aggregate-report
```

Generated artifacts are organized as follows:

```text
analysis/reports/spatial-sequence/
  report.md
  convergence.json
  figures/

output/spatial-sequence/
  runs/
  logs/
  videos/
```

The Markdown report and static figures follow the repository's analysis-report
convention. Checkpoints, predictions, logs, and MP4 files remain under the ignored
`output/` tree.
