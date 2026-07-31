# Superseded mint + lavender T-raster pilot 01

This folder preserves the first moving two-source spatial pilot for provenance.
It is no longer part of the active analysis narrative; the parallel-strip
session `two_horizontal_raster_mint_lavender_01` replaced it as Notebook 08's
current experiment.

- trial ID: `t_raster_mint_lavender_01`;
- session: `cyranose_reading_pose_session_20260727_175116`;
- horizontal top strip: mint;
- vertical lower strip: lavender;
- the paper sources did not touch.

## Reported paper geometry

The paper towel is treated as a 26.5 x 26.5 cm local surface with its origin at
the top-left:

| Source | Paper X | Paper Y |
|---|---:|---:|
| Mint | 4.5-22.0 cm | 4.0-6.0 cm |
| Lavender | 11.0-13.0 cm | 8.0-24.0 cm |

Notebook 08 uses the approximate manual registration
`paper_x = 40 - desk_x` and `paper_y = desk_y - 20`. The source rectangles are
reference overlays and evaluation regions only; they do not alter model scores,
spatial support, smoothing, or thresholds.

## Acquisition QC

- 452 Cyranose readings were recorded.
- 452/452 readings matched a RealSense pose within 250 ms.
- Absolute pose/readout offset was 57.2 ms median and 97.0 ms p95.
- 381 readings occurred during active flag 2.
- After causal-history and paper-boundary checks, 338 readings remain at the
  2.0 s working pose correction.
- 162/338 paper-local readings pass the strict 1.0-3.5 cm snout-height band;
  335/338 are eligible for the explicitly labeled 0.5-3.5 cm soft-height view.

## Frozen-model result

The original executed notebook and HTML report are preserved directly in this
folder. They load and verify these artifacts without retraining:

- `mint-lavender-ambient-v1`;
- `temporal-mint-seeker-v1`.

The current result is a useful negative/diagnostic pilot, not a recovered T:

- the pure-odor hierarchy is strongly mint-dominant over much of the moving
  scan;
- median strict background mint affinity is 0.998 and 64.2% of strict
  background readings exceed 0.5;
- mint source-region ranking AUC is 0.589;
- no strict-height reading falls inside the narrow reported lavender rectangle;
- the closest strict lavender sample is 0.42 cm away, 17 readings fall within
  1 cm, and their median lavender affinity is 0.000;
- smoothed mint, lavender, and ambient maps therefore must not be interpreted
  as successful mixture decomposition or concentration recovery.

The notebook keeps the continuous mint/lavender maps, a combined color
composition, a secondary uncertainty view, frozen mint active/recovery
diagnostics, raw sampling counts, and cross-lag support. The reported source
shapes remain visible as translucent overlays so a visually attractive map
cannot substitute for the quantitative checks.
