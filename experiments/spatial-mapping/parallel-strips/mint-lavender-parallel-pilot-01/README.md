# Mint + lavender parallel-strip pilot 01

This is the active two-source moving-raster pilot:

- trial ID: `two_horizontal_raster_mint_lavender_01`;
- session: `cyranose_reading_pose_session_20260727_184252`;
- physical layout: horizontal mint strip above horizontal lavender strip;
- both strips were approximately 17 cm long and 3 cm wide;
- the reported clean gap was approximately 9 cm;
- the sources did not touch.

## Paper-local geometry

Notebook 08 displays the 26.5 x 26.5 cm paper with its origin at the top-left.
The operator described the narrow top/bottom ranges as X and the common long
range as Y. Because the notebook uses conventional `X=left-right` and
`Y=top-bottom`, those reported ranges are transposed for plotting:

| Source | Paper X | Paper Y |
|---|---:|---:|
| Mint | 5.0-22.0 cm | 6.0-8.0 cm |
| Lavender | 5.0-22.0 cm | 18.5-21.0 cm |

The manual desk registration is `paper_x = 40 - desk_x` and
`paper_y = 46.5 - desk_y`. The Paper Y reflection was fixed from the
operator-confirmed acquisition order—bottom-left start, first pass upward,
then right and downward—not from odor scores. A stationary pre-raster dwell is
excluded; the analyzed raster begins at 40.0 s at the lower-left departure.
Source rectangles are reference overlays and evaluation regions only; they do
not alter scores, thresholds, smoothing, lag selection, or spatial support.

## Acquisition QC

- 296 Cyranose readings were recorded.
- 296/296 readings matched a RealSense pose within 250 ms.
- Absolute pose/readout offset was 58.1 ms median and 96.3 ms p95.
- 266 readings occurred during active flag 2.
- The 2.0 s paper-local view retains 232 causal-history readings after the
  stationary pre-raster dwell is excluded.
- 114/232 pass strict 1.0-3.5 cm height QC.
- 230/232 are eligible for the labeled 0.5-3.5 cm soft-height view.
- Lag-consensus support covers 2708/2916 grid cells (92.9%).

## Frozen-model result

[`analysis/notebooks/08_mint_lavender_parallel_raster.ipynb`](../../../../analysis/notebooks/08_mint_lavender_parallel_raster.ipynb)
loads and checksum-verifies the two existing artifacts:

- `mint-lavender-ambient-v1`;
- `temporal-mint-seeker-v1`.

It also trains a three-state lavender model using stationary files only,
freezes it as `temporal-lavender-seeker-v1`, and reload-verifies SHA-256
`c11d837492b30c2d0c824826c3819749756a73a9698c1992f54ef9d7ff65228a`.
The raster is not used for feature or regularization selection. Complete-trial
validation is 81.2% balanced accuracy, with active/fading lavender AUC
0.968/0.929; the current-only snapshot is slightly better at 84.0% balanced
accuracy.

The simpler geometry fixes the previous source-sampling imbalance: strict
working-lag data include 5 readings inside the mint rectangle and 9 inside the
lavender rectangle. The 1 cm vicinity contains 13 and 17 readings,
respectively.

The model result remains mixed:

- raw mint source-ranking AUC is 0.705;
- frozen temporal active-mint source-ranking AUC is 0.963;
- strict-background readings above the 0.5 mint boundary decrease from 69.0%
  for raw mint affinity to 28.0% after temporal logic;
- lavender current-identity source-ranking AUC is 0.776, but symmetric
  temporal active-lavender ranking falls to 0.648;
- readings inside the reported lavender strip are predominantly mint-like.
- on the unsmoothed first downward lane, the gap has median raw lavender
  allocation 0.535 and maximum 0.896 before the snout reaches lavender.
  Temporal active-lavender evidence falls to median 0.193 and fading-lavender
  evidence to 0.006, showing that the original purple gap was chiefly a
  closed-panel classifier/domain-transfer error;
- direction-aware post-versus-pre fading AUC is 0.296 for mint and 0.290 for
  lavender. Neither fading map is validated as a post-source recovery map on
  this moving raster.

This is a partial mint result and a negative lavender moving-transfer result.
It is not successful two-odor localization, a concentration map, or evidence
of zero-shot mixture decomposition. Notebook 08 now shows continuous
closed-panel allocations plus symmetric active mint, fading mint, active
lavender, and fading lavender heatmaps. Its combined presentation map uses
green for exclusive active-mint evidence, purple for exclusive active-lavender
evidence, and red for their shared/ambiguous component. It preserves
continuous sub-threshold evidence. The four-panel diagnostic now displays the
continuous active-lavender consensus on a labeled 0-0.5 scale (observed
maximum 0.441) and fading lavender on its own observed 0-0.1167 scale.
This reveals weak nonzero evidence without implying that it is as strong as
the mint panels, which retain 0-1 scales. Red is not evidence of a chemical
mixture. The former categorical
temporal-context label panel was removed because it hid uncertainty and forced
a winner. The lane audit is the primary evidence for chronology; the smoothed
field alone cannot establish whether a score occurred before or after a source
crossing.
