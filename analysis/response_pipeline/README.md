# Response Pipeline

Shared primitives for turning one recorded Cyranose + RealSense session into
spatially-located odor-response readings. This package replaces the
`load_trial` / `interpolate_pose` / `smooth_grid`-style code that was
previously redefined from scratch inside notebooks 03 through 09.

## Which notebooks use this, and how much

- **03, 04** import the full dict-based pipeline below (`qualifying_rows`,
  `smooth_grid`, etc.). 04 was the pilot; 03 only migrated its baseline/RMS
  computation, because its scan-window filter runs on the raw
  `pcnose_device_time_s` clock rather than `pose_elapsed_s` — a pre-dates-the-
  fix quirk from being the first raster notebook — so its lag/pose logic
  stayed local rather than being forced into `TrialConfig`'s pose-window
  model.
- **05, 06, 08, 09** are a different lineage: pandas/numpy classifier-
  application notebooks (`pd.read_csv` + a frozen model's `.predict_proba`),
  not the dict-based raster pattern. Only their baseline/normalize step
  (`compute_baseline_array`, `normalized_channels` — the array-oriented
  equivalents of `compute_baseline`/`sensor_response` below) was migrated;
  everything downstream — classifier scoring, pose lag-shift/height/bounds
  QC, and especially the temporal-feature-engineering functions
  (`irregular_ema`, `interpolate_vectors`, `build_temporal_features`) that
  feed each notebook's *training* data — stays local. That training-data ETL
  in particular was deliberately left untouched: it feeds frozen, checksummed
  model bundles, and reordering its floating-point math is the one place that
  could theoretically nudge a retrained model's coefficients.
- **01, 02** are untouched. 01 is early/exploratory. 02 is the notebook that
  *derives* `lag_s` in the first place, so it isn't "consuming" a fixed value
  the way the others are.

## Pipeline stages (dict-based raster path: 03, 04)

1. **Load** (`io.py`) — read a session's `cyranose_reading_pose.csv`,
   `session_metadata.json`, and `alignment_summary.json`.
2. **Baseline** (`response.py: compute_baseline`) — the median of the back
   half of the clean-air/purge phase (`pcnose_flag == "1"` by default), per
   sensor channel. The back half is used so the reference reflects the
   sensor once it has settled into that phase.
3. **Response** (`response.py: sensor_response`) — each channel's percent
   change from baseline, combined by root-mean-square into one scalar per
   reading. RMS is used (rather than a plain mean) so that channels moving in
   opposite directions don't cancel out.
4. **Lag-shifted pose lookup** (`pose.py: qualifying_rows`) — the sensor does
   not react instantly, so a reading at time T is paired with the snout's
   interpolated position `lag_s` seconds earlier, not its position at T.
   `lag_s` must be measured for the physical rig first (see
   `analysis/notebooks/02_response_lag_validation.ipynb`); it is not derived
   automatically.
5. **Height and desk-bounds QC** (`pose.py: qualifying_rows`) — a reading is
   only trusted for a given (x, y) if the snout was within `height_band_cm`
   of the desk and inside `desk_bounds_cm` at the lag-corrected time.
6. **Spatial smoothing** (`smoothing.py: smooth_grid`) — Gaussian-weighted
   average of retained readings onto a fixed grid, with a minimum
   support-count per cell so no cell is interpolated from a single reading.

## Array-based path (classifier-application notebooks: 05, 06, 08, 09)

`compute_baseline_array(resistances, flags, baseline_flag)` and
`normalized_channels(resistances, baseline)` are the same baseline/RMS-input
formula as `compute_baseline`/`sensor_response` above, but operating on whole
`(rows, channels)` numpy arrays instead of one dict-per-row — this is the
shape these notebooks already work in (`pandas.DataFrame.to_numpy()`), and it
was the single most duplicated block in the project: the identical three-line
pattern appeared independently in nine places across four notebooks before
this migration.

## What's configured per trial, and what isn't

`TrialConfig` (`config.py`) holds everything that varies by physical setup or
recording session: `lag_s`, `height_band_cm`, `desk_bounds_cm`,
`pose_window_s`, and the sensor/flag column naming.

`lag_s` and `desk_bounds_cm` are never derived automatically — they come from
a dedicated lag-calibration trial and from the desk-tag calibration
(`calibration/`) respectively. Everything else in `TrialConfig` defaults to
this repository's existing rig and recorder
(`record_cyranose_reading_pose.py`); the CSV column names themselves
(`pose_elapsed_s`, `snout_desk_x_cm`, ...) are fixed by that recorder's output
format and are not configurable, since any session recorded with it has the
same column shape regardless of odor or desk layout.

## Testing

`test_response_pipeline.py` (repository root) has two kinds of tests:

- Unit tests with small synthetic data and hand-computed expected values, for
  each pure function.
- Regression tests that load two real, already-reviewed sessions
  (`line_raster_blank_fresh_01` and `line_raster_mint_blotter_retry_01`) and
  assert against the exact numbers notebook 04 already reports in its saved
  output, so a refactor here cannot silently change a published result.
