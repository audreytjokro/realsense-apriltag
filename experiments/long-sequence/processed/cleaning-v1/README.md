# Long-sequence cleaning v1

This directory contains the first frozen, auditable processing export for the
11 reviewed July 30 and August 3 long recordings. Raw session CSV and MP4 files
are never rewritten.

## Files

- `cleaning_schema_v1.json`: versioned preprocessing, geometry, eligibility,
  and pilot-model rules;
- `long_sequence_processed_v1.csv.gz`: all 11,082 raw rows with lineage,
  32 raw resistance channels, 32 baseline-normalized channels, 32 causal model
  channels, synchronized and lag-corrected pose fields, source labels, and
  explicit eligibility/exclusion fields;
- `run_manifest.csv`: one row per recording with raw SHA-256, baseline values,
  source counts, eligibility counts, and the primary train/holdout role;
- `sequence_source_classifier_pilot_v1.joblib`: causal temporal pilot trained
  on ten complete recordings, excluding `20260803_170357`;
- `primary_holdout_predictions.csv` and `primary_holdout_metrics.json`: output
  for the untouched parallel-strip recording;
- `leave_one_recording_out_metrics.csv`: fixed-spec whole-recording robustness
  audit.

## Interpretation

The primary holdout improves from 0.537 macro recall with only the current
32-channel response to 0.663 with causal temporal features. Recall is 0.500
for background, 0.639 for mint, and 0.851 for lavender. Across the nine
three-class whole-recording folds, temporal median macro recall is 0.516.

This is an encouraging pilot, not a final odor model. There are only 11
independent recordings, rows inside a recording are autocorrelated, and target
labels are protocol-approximate source regions rather than direct chemical
measurements. The model artifact is saved for reproducibility but is not
declared scientifically validated or session-invariant.

Rerun `analysis/notebooks/11_cleaning_schema_and_sequence_holdout.ipynb` from
the repository root to regenerate and validate the export.
