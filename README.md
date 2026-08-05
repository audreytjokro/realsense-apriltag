# RealSense + Cyranose spatial odor mapping

This repository combines two measurements:

- an Intel RealSense camera estimates the Cyranose snout position from AprilTags;
- a Cyranose supplies 32-channel chemical-sensor readings over time.

The working goal is to associate each odor reading with a defensible desk
position, then recover simple spatial odor patterns such as a line, circle, or
letter. The repository structure reflects the experiments completed so far,
but the stage names are intentionally descriptive rather than permanent. New
stages can be added without renaming the earlier data.

## Current status

- Camera, tagged-cube, desk-tag, and snout-tip calibration are working.
- Each Cyranose reading is paired with the nearest camera pose.
- Pose matches farther than 250 ms are retained but marked invalid and receive
  no pose coordinates.
- The dedicated response-lag trials originally supported about 3.0 seconds:
  the two primary pairs estimated 2.90 s and 2.79 s. Those historical analyses
  remain unchanged.
- Notebook 06 now audits that assumption on Horizontal line raster-454 without
  retraining either classifier. The same 157 readings are remapped over
  0-5 seconds in 0.25-second steps and scored numerically against the fixed
  reported strip before gridding or smoothing. The active-score-weighted
  distance criterion reaches its minimum at **2.25 s**, while distance and
  source-ranking AUC jointly support a provisional **2.25-2.50 s** band.
  Separate upward/downward minima are 2.50 s and 2.25 s. Because the strip
  coordinates are approximate and this run supplies the calibration target,
  this narrows the plausible range but does not establish a universal lag or
  replace independent validation. Notebook 06 therefore uses **2.0 seconds as
  its working spatial correction**: a directly tested, conservative compromise
  between 1.5 and 2.25 seconds that reduces the visible 3-second overcorrection
  without claiming the approximate ground truth supports millisecond precision.
  The reported source coordinates are fixed independently of lag so the target
  cannot move when this setting changes.
- Notebook 04 now compares `line_raster_mint_blotter_retry_01` against the
  existing clean background `line_raster_blank_fresh_01`. Digital alignment is
  excellent for both runs. After 3.0 s lag correction, the blotter retry has
  3.99 times the background run's median RMS response.
- Notebook 05 adds a documented 70-trial classifier pilot: 15 blank, 15 mint,
  20 bergamot, and 20 lemongrass PCnose+ trials. It validates row/window models
  by holding out complete trials while mixing recording dates normally.
- The two-stage regularized model achieves grouped trial AUC 1.000 for odor
  presence and 0.995 for mint versus the recorded bergamot/lemongrass panel.
  A three-odor held-out-trial check is 92.7% accurate. The tiny MLP did not
  improve trial AUC, so logistic regression remains the selected pilot model.
- Notebook 07 adds a separate balanced stationary panel: 15 ambient, 15 mint,
  and 15 lavender PCnose+ trials. All 45 repository copies match their source
  hashes, contain 32 channels, and contain 58 active-exposure rows. Three
  nonfatal phase-count deviations are retained as explicit manifest warnings.
- Its primary hierarchy first detects ambient versus odor, then distinguishes
  mint from lavender using direction-normalized fingerprints. Five-fold
  validation holds out complete trials and achieves 100.0% balanced accuracy:
  ambient, mint, and lavender recall are each 15/15. With only 15 trials per
  class, each perfect recall has an approximate 79.6-100.0% Wilson interval.
  The simpler direct three-class diagnostic reaches 86.7% balanced accuracy
  (ambient 15/15, mint 12/15, lavender 12/15).
- The lavender result is recording-session limited: lavender was collected on
  July 27, while the existing mint and ambient trials were recorded July 22-23.
  Trial-held-out validation prevents row leakage but cannot separate lavender
  identity from date-dependent sensor state. The frozen
  `mint-lavender-ambient-v1` bundle therefore represents closed-panel
  separability, not session-invariant recognition, moving-scan transfer, or
  mixture decomposition.
- Notebook 08 now performs the current moving two-source transfer test while
  hash-verifying the existing identity and active-mint artifacts and adding a
  separately frozen three-state lavender temporal model. The reviewed raster
  uses two parallel,
  physically horizontal strips on a 26.5 x 26.5 cm paper-local surface:
  mint at X 5-22, Y 6-8 cm and lavender at X 5-22, Y 18.5-21 cm. Digital
  synchronization is excellent (296/296 matches; 96.3 ms p95).
- Its paper trace is oriented from the confirmed operator sequence rather than
  odor scores: `paper_x = 40 - desk_x`, `paper_y = 46.5 - desk_y`. The
  stationary pre-raster dwell is excluded at 40.0 s, so the displayed Start is
  bottom-left, followed by an upward pass and a downward return.
- The new lavender model distinguishes **active lavender**, **fading
  lavender**, and **no active lavender** using causal history from the 45
  stationary trials. Trial-grouped validation passes the predeclared pilot
  gate: temporal balanced accuracy is 81.2%, active-lavender AUC is 0.968,
  fading-lavender AUC is 0.929, and per-state recall is 87.0%, 71.0%, and
  85.7%, respectively. The current-only snapshot is slightly better on
  balanced accuracy (84.0%), so temporal context is retained for its causal
  state interpretation rather than claimed as a universal accuracy gain.
- An unsmoothed first-downward-lane audit explains the apparent lavender plume
  before the physical lavender strip. In the clean gap, the closed-panel
  identity model assigns a median lavender share of 0.535 (maximum 0.896)
  before lavender has been encountered; the temporal model lowers the median
  active-lavender evidence to 0.193 and fading evidence to 0.006. The original
  purple gap is therefore chiefly a model-allocation/domain-transfer error,
  not proof that the snout had already sampled lavender.
- The final symmetric maps show active and fading evidence for both odors. A
  separate combined continuous storytelling map encodes exclusive active mint
  in green, exclusive active lavender in purple, and the shared component
  `min(active mint, active lavender)` in red. Red therefore marks model
  overlap or ambiguity, not a proven chemical mixture. It preserves
  sub-threshold evidence. The four-panel diagnostic now also retains the
  continuous active-lavender consensus and plots it on an explicitly labeled
  0-0.5 scale (observed maximum 0.441). Fading lavender is shown on its own
  observed 0-0.1167 scale. These expanded diagnostic scales make weak
  nonzero evidence visible, but their brightness must not be compared directly
  with the mint panels' 0-1 scales. Raw mint
  source-ranking is AUC 0.705, while the presence-gated active-mint
  pipeline reaches 0.963. The analogous identity-confirmed active-lavender
  score reaches only 0.648 and remains near zero on the actual lavender strip,
  which is still strongly mint-like. Direction-aware fading audits are also
  negative (post-source versus pre-source AUC 0.296 for mint and 0.290 for
  lavender), so the fading maps are diagnostics rather than validated recovery
  localization. This is a partial mint result and a negative lavender/fading
  moving-transfer result—not recovered two-odor localization.
- Notebook 09 adds five longer approximately random-motion sequences:
  mint-only, lavender-only, both caret odor assignments, and an inverted caret.
  All five promoted runs have 100% digital pose/readout matching and complete
  32-channel sensor vectors. The latest inverted-caret run contains 1,114
  readings over 619.3 host-timed seconds.
- The frozen models and fixed mapping rules were applied without retraining.
  The mint-only horizontal run preserves the clearest spatial ranking
  (identity-independent RMS AUC 0.858; active-mint AUC 0.639). Across all
  odor/source comparisons, median current-identity and active-state AUC is
  0.523/0.524, while median identity-independent odor AUC is 0.646. The long
  caret and V runs therefore preserve some broad odor structure but do not
  cleanly recover source geometry or mint-versus-lavender identity.
- The first four July 30 guide alerts occurred early because the old guide
  treated a fast Cyranose device counter as wall-clock seconds. The recorder
  files themselves remain valid. The guide now derives elapsed time from the
  host midpoint timestamp; the latest run confirms the corrected behavior.
- The August 3 long-sequence batch adds six reviewed recordings: two X-shape
  runs with mint on the backslash and lavender on the forward slash, two
  90-degree-rotated X replicates, one inverted-caret run with lavender on the
  left and mint on the right, and one parallel-strip run with mint above
  lavender. Their host-timed durations range from 10.36 to 11.17 minutes and
  contain 1,105-1,179 flag-2 readings apiece.
- All six August 3 recordings have complete 32-channel sensor vectors and
  100% pose/readout matching inside the 250 ms threshold; p95 absolute timing
  offsets are 91.4-95.7 ms. After the existing 2.0 s spatial correction,
  paper-boundary check, and 1.0-3.5 cm height filter, 52.3-91.9% of flag-2
  readings remain. The first X run is retained with a low-height-coverage QC
  note. The first rotated-X run contains one isolated S11 acquisition spike,
  which should remain in the raw file but be masked or median-filtered during
  analysis. These are usable sequence recordings, not yet evidence that the
  corresponding source geometries were reconstructed successfully.
- The six August 3 MP4 files decode without errors but are time-compressed:
  they play for approximately 3.85-4.30 minutes while their CSV host
  timestamps span 10.36-11.17 minutes. The writer labels the video with the
  camera's nominal frame rate while the pose-processing loop supplies frames
  more slowly. Use the CSV host timestamps for sensor-pose analysis and do not
  align odor readings from MP4 playback time.
- Notebook 10 audits all 11 reviewed July 30 and August 3 long sequences with
  one read-only QC policy. All 11 pass the analysis-readiness gate: every CSV
  has complete 32-channel rows, at least 99% accepted smell/pose matches, and
  158-182 occupied 2 cm cells after the strict spatial rules. The first August
  X run is retained with a height-coverage caution, and the 15:44:20 X run has
  one isolated acquisition row that is flagged and causally median-filtered.
  Six-run trajectories are colored green/yellow/red for early/middle/late
  time, and 32-sensor heatmaps show exact per-channel contributions to the
  frozen mint-versus-lavender identity logit. An exploratory transition audit
  is strongly mint-dominant and stable to 0.5-1.5 cm source-mask margins, but
  it is explicitly treated as possible sensor/model carryover rather than
  proof of physical odor transfer.
- Applied to the existing rasters, median combined mint score is 0.000 for the
  clean blank and 0.805 for the blotter mint run; 77.0% of retained mint rows
  exceed the working 0.5 score threshold. These are uncalibrated pilot scores,
  not universal mint probabilities or geometry validation.
- Notebook 05 also applies the same model to the older rectangle and two
  paper-towel line scans. The rectangle has high odor-presence score (0.992)
  but low mint-identity score (0.191), exposing a stationary-to-moving
  domain-shift problem. The first paper-towel line has some near-source
  contrast (0.920 near versus 0.736 far), but far-field scores remain too high
  for crisp geometry. The fresh-towel score is substantially structured by
  scan time (Spearman rho +0.60), consistent with response persistence.
- Notebook 05 now includes the day-old dry-strip trial
  `line_raster_mint_dry_strip_01`. Digital alignment remained strong
  (284/286 matches; 101.1 ms p95), but only 69/174 reviewed raster readings
  were inside the 1.0-3.5 cm height band. The raster mint score was effectively
  zero and source-region ranking was chance-like (AUC 0.490), so this run is
  retained as a source-loading diagnostic rather than replacing the earlier
  line panel.
- Notebook 05 also reviews **Horizontal line raster-454**
  (`line_raster_mint_moderate_strip_01`): a fresh 17 x 3 cm strip with exactly
  three drops and a five-minute wait. Digital alignment is excellent
  (238/238 matches; 95.2 ms p95). Moderate loading restores a strong moving-scan
  response and approximate source-region ranking (median 0.992 inside versus
  0.811 outside; AUC 0.835), but 55.1% of outside readings still exceed 0.5 and
  score remains correlated with scan time (rho +0.51). It is the strongest
  source-loading/detectability pilot so far, not a clean recovered 3 cm line.
- Notebook 06 adds a three-state temporal pilot: **active mint**, **mint
  recovery**, and **no active mint**. Model and feature selection use only the
  70 stationary PCnose+ trials, with complete trials kept together in nested
  grouped validation; the moving raster is not used for tuning.
- Temporal history improves held-out-trial balanced accuracy from **83.9% to
  87.7%**. With Notebook 06's 2.0-second working correction and fixed reported
  source coordinates, active-mint source-ranking AUC improves from **0.672 to
  0.938**, while outside-source readings above the working 0.5 threshold
  decrease from **60.5% to 22.2%** after the required odor-presence gate. The
  gated score-time correlation changes from **+0.53 to +0.28**, so some
  scan-order dependence remains. This is promising spatial structure, not a
  final calibrated probability or recovered-width claim.
- Notebook 06 now provides a four-chart presentation sequence: tracked raster
  before temporal logic, tracked raster after temporal logic, softly
  height-weighted heatmap before temporal logic, and the agreement-weighted
  lag-consensus heatmap after temporal logic. Light blue consistently marks
  current/active mint, light green marks mint recovery, and magenta marks snout
  height on the shared time chart. The heatmaps share the same grid, spatial
  kernel, height weighting, and support rule; the after-temporal map additionally
  requires stability across the four documented lag corrections.
- Its lag-sensitivity section adds a predeclared numerical audit and matched
  0.00, 1.50, 2.25, and 3.00 s heatmaps. These maps are diagnostic views of the
  same frozen scores; lag selection is based on raw distance to the reported
  strip rather than visual attractiveness. The presentation heatmaps and all
  dependent Notebook 06 spatial diagnostics are now rerun consistently with
  the documented 2.0-second working correction.
- A final timing-ceiling check combines the predeclared 1.50, 1.75, 2.00, and
  2.25 s maps without using the reported rectangle as a mask or shape prior.
  The cell-wise median changes the 2.0 s map very little. Requiring agreement
  across lags reduces the above-strip extension from 7.3 to 6.3 cm and the
  fraction of thresholded grid cells outside the approximate strip from 70.8%
  to 66.7%, but does not recover a rectangle. This is evidence that selecting
  another fixed lag is near its useful ceiling for this run. The presentation
  output shows the agreement-weighted consensus alone; the four component
  calculations remain reproducible inside the notebook.
- A secondary height-weighted presentation map preserves the strict QC result
  while softly incorporating near-threshold 0.5-1.0 cm readings. It retains
  the 3 cm spatial radius and two-reading minimum and reports the supported
  grid-cell count; it is not treated as a replacement for strict QC.
- The selected deployment bundle is frozen as
  `temporal-mint-seeker-v1`. It contains the odor-presence, mint-identity, and
  three-state temporal models plus their preprocessing configuration,
  training-data digests, model card, and checksum. Notebook 06 reload-tests the
  saved artifact against the in-memory raster scores. Future shape experiments
  must load this version unchanged; retraining requires a new version.
- Deployment uses a required hierarchical presence gate: temporal active-mint
  evidence is retained only when the frozen odor-presence model scores at least
  0.5, and is otherwise set to zero before spatial smoothing. Recovery remains
  ungated. This removes 34/34 known-clean startup false positives without
  changing the fitted models or their checksum. In the presentation figures,
  this changes the after-temporal tracked raster, after-temporal heatmap, and
  active-mint diagnostics; the before-temporal figures and recovery evidence
  are intentionally unchanged. The after-temporal heatmap's dashed contour is
  the working 0.5 detection boundary, so faint sub-threshold interpolation
  outside it is context rather than a positive detection.
- Mint-source minus background is positive in 96.3% of shared grid cells. Its
  row-wise profile peaks near Desk Y = 34 cm with an apparent 14 cm
  half-maximum width—much broader than the reported 4 cm paper source. The
  exact oil dose is unknown and was reported excessive, so this run emphasizes
  detectability rather than concentration or edge sharpness.
- The first attempted blank raster also produced a band, but the blank towel
  was placed over the previous mint location. That trial is preserved as a
  contamination-suspected diagnostic, not a clean control.
- The retry improved the median mint hover height to 2.56 cm, but the manual
  height problem remains localized around the source: 21/147 in-bounds mint
  readings were above the 3.5 cm analysis limit, spanning Desk Y 31.2–43.8 cm.
  The clean background has no height exclusions and a 1.84 cm median.
- Overall coverage is high—91.3% of the predefined grid has common support—but
  that summary hides the localized high-snout gap. Smoothing supplies values
  around the missing region; it cannot recover the omitted source-crossing
  measurements.
- Geometry remains analysis-sensitive. The hotspot angle changes from about
  3 degrees at 2.0 s lag to 20 degrees at 3.0 s and 57 degrees at 4.0 s. Across
  smoothing radii it ranges from -27 degrees to +20 degrees; the 2 cm result is
  nearly horizontal but not stable enough to select as the answer. Scan
  progress remains almost perfectly confounded with Desk X.

These are preliminary experimental results, not a claim that arbitrary odor
shapes can already be reconstructed reliably.

## Repository map

```text
calibration/                     Calibration workflow, modules, and data
experiments/
  pose-tracking/                 Early AprilTag and snout-pose recordings
  exploratory-mapping/          Early spatial odor-map recordings
  classifier/                    Stationary odor trials for identity models
  response-lag/
    lag-tests-valid/             Blank control and primary lag pairs
    lag-tests-excluded/          Preserved attempts excluded from the estimate
  spatial-mapping/
    line-raster/matched-pairs/   Reviewed blank/mint blocks kept together
    line-raster/background-referenced/ Mint-source retries using an earlier clean background
    line-raster/pilot-usable/    Reviewed raster pilots worth analyzing
    line-raster/source-loading-diagnostics/ Dry or otherwise altered source-strength checks
    line-raster/horizontal-line-raster-454/ Current three-drop horizontal-line pilot
    line-raster/excluded-controls/ Preserved controls excluded from primary comparison
    parallel-strips/mint-lavender-parallel-pilot-01/ Current two-source transfer pilot
    superseded-pilots/mint-lavender-t-pilot-01/ Preserved earlier T diagnostic
  long-sequence/
    2026-07-30_batch-01/       July 30 random-motion batch
    2026-08-03_batch-01/       August 3 long-sequence batch
analysis/
  notebooks/                     Primary analysis and visualization files
  reports/                       Optional notebook exports for easy viewing
  figures/                       Diagnostic images and exported figures
  protocols/                     Experiment instructions and design notes
  manifests/                     Templates and analysis-side indexes
record_cyranose_reading_pose.py  Main synchronized recorder
random_waypoint_guide.py         Optional waypoint suggestions and run timer
track_calibrated_cyranose_pose.py Live calibrated pose display
pcnose_serial.py                 Direct Cyranose serial protocol support
cyranose_reading_pose_session_*/ Newly recorded raw sessions pending promotion
```

The complete recording index is in
[`experiments/manifest.md`](experiments/manifest.md). A manifest is simply a
catalog that says what each timestamped folder contains, whether it is used in
the primary analysis, and why.

## Typical commands

Run commands from the repository root.

### Calibration

```powershell
python calibration/calibrate.py
```

Calibration data is stored under `calibration/data/`. Older snout-tip attempts
are archived under `calibration/archive/`; the active calibration files remain
in their expected locations.

### Live calibrated pose

```powershell
python track_calibrated_cyranose_pose.py
```

### Record synchronized Cyranose readings and pose

Close PCnose+ first so that it releases the serial port, then run:

```powershell
python record_cyranose_reading_pose.py --port COM4 --baud 57600 --interval 0.2 --max-sync-ms 250 --save-video --trial-id TRIAL_ID --trial-label TRIAL_LABEL --notes "NOTES"
```

Saved MP4 files use H.264 (`libx264`, CRF 18, `yuv420p`) through the `ffmpeg`
executable on `PATH`. Recording fails explicitly if FFmpeg is unavailable; it
does not fall back to the older MPEG-4 Part 2 codec.

The recorder always creates a new timestamped session at the repository root.
After a session is reviewed, place it in the appropriate `experiments/` stage
and add it to the manifest. This keeps collection simple while allowing the
research organization to evolve.

### Optional random-waypoint guide

Start the synchronized recorder first. In a second PowerShell terminal, launch
the guide with the exact same trial ID:

```powershell
python random_waypoint_guide.py --trial-id TRIAL_ID --seed 73001 --duration-s 600 --width-cm 26.5 --height-cm 26.5 --margin-cm 1.5
```

The guide finds the newest recent root session with that trial ID and saves
`waypoint_sequence.csv` plus `waypoint_metadata.json` inside it. Space or `N`
advances the optional suggestion. `Q`, Escape, or closing the guide window
stops only the guide. The displayed 600-second alert does not stop the
Cyranose + RealSense recorder; stop that recorder manually in its own camera
window. Long random-motion sessions belong under
`experiments/long-sequence/` after review.

The elapsed display uses the recorder's host midpoint timestamp, not
`pcnose_device_time_s`. Older July 30 waypoint logs still have valid
`host_timestamp_utc` entries, but their legacy `recorder_elapsed_s` values are
about 1.6 times faster than physical time.

### Open or rerun the analysis notebooks

```powershell
jupyter lab analysis/notebooks/02_response_lag_validation.ipynb
jupyter lab analysis/notebooks/03_line_raster_mint_01.ipynb
jupyter lab analysis/notebooks/04_mint_vs_blank_raster.ipynb
jupyter lab analysis/notebooks/05_mint_identity_classifier.ipynb
jupyter lab analysis/notebooks/06_dynamic_mint_exposure_recovery.ipynb
jupyter lab analysis/notebooks/07_mint_lavender_ambient_classifier.ipynb
jupyter lab analysis/notebooks/08_mint_lavender_parallel_raster.ipynb
jupyter lab analysis/notebooks/09_long_random_sequence_shape_maps.ipynb
jupyter lab analysis/notebooks/10_long_sequence_qc_transfer_and_32d.ipynb
jupyter lab analysis/notebooks/11_cleaning_schema_and_sequence_holdout.ipynb
```

The numbered notebooks are saved with their outputs, so they can be inspected without
rerunning the cells. Optional HTML exports under `analysis/reports/` can be
opened in a browser without Jupyter.

If Jupyter reports that a module compiled for NumPy 1.x cannot run with NumPy
2.x, restart the notebook kernel and run all cells. Notebooks 05 and 07 remove
the roaming user-site package directory before importing NumPy. Alternatively,
close Jupyter and launch it from PowerShell with:

```powershell
$env:PYTHONNOUSERSITE = "1"
jupyter lab analysis/notebooks/07_mint_lavender_ambient_classifier.ipynb
```

## Timing and Cyranose flags

- `pose_elapsed_s` and the serial-request midpoint are the wall-clock timing
  basis for pose alignment and physical response-lag correction.
- `pcnose_device_time_s` is useful for inspecting the device sequence, but in
  these recordings it did not advance at wall-clock speed. Do not interpret it
  as physical seconds.
- MP4 playback duration is not a reliable wall-clock basis in recordings made
  by the current synchronous video-writing loop. A video may contain frames
  from the full session while playing substantially faster than real time.
  Use the host-timestamped CSV pose rows for temporal alignment.
- Flag 0 is the idle/pre-run state observed in this workflow.
- Flag 1 is the purge/reference phase.
- Flag 2 is the sample-intake/measurement phase.

For current analyses, use the latter half of flag 1 as the per-run reference.

## Data handling conventions

- `valid` means suitable for the stated primary analysis.
- `excluded` does not mean corrupt. Excluded recordings are preserved for
  diagnostics, alternative analyses, or auditability.
- Never combine sessions only because their filenames are close in time. Use
  trial metadata and the manifest.
- Do not move the fixed desk tag during a group of spatial trials.
- Record source geometry, scan direction, approximate height, and relevant
  setup changes in `--notes`.

## Analysis folders

- [`analysis/notebooks/01_trajectory_odor_maps.ipynb`](analysis/notebooks/01_trajectory_odor_maps.ipynb)
  is the exploratory trajectory/odor-map notebook.
- [`analysis/notebooks/02_response_lag_validation.ipynb`](analysis/notebooks/02_response_lag_validation.ipynb)
  estimates the physical odor-response lag from the accepted forward/reverse
  trials.
- [`analysis/notebooks/03_line_raster_mint_01.ipynb`](analysis/notebooks/03_line_raster_mint_01.ipynb)
  applies the 3.0 s correction and visualizes the first controlled raster scan.
- [`analysis/notebooks/04_mint_vs_blank_raster.ipynb`](analysis/notebooks/04_mint_vs_blank_raster.ipynb)
  is the current blotter mint-source versus clean-background analysis. It
  applies the 3.0 s correction, uses shared spatial support, directly subtracts
  background from mint-source, and includes height, coverage, scan-time,
  smoothing, lag, spatial-profile, and 32-sensor diagnostics.
- [`analysis/notebooks/05_mint_identity_classifier.ipynb`](analysis/notebooks/05_mint_identity_classifier.ipynb)
  trains a two-stage blank-versus-odor and mint-versus-recorded-odors model,
  validates it with whole trials held out, compares regularized logistic
  regression with a tiny MLP, applies the resulting mint score to the existing
  rasters, and compares dry versus measured moderate source loading.
- [`analysis/notebooks/06_dynamic_mint_exposure_recovery.ipynb`](analysis/notebooks/06_dynamic_mint_exposure_recovery.ipynb)
  compares current-only and causal temporal classifiers using nested,
  trial-grouped validation, locks the selected temporal model without using
  raster outcomes, and applies separate active-mint and recovery scores to
  Horizontal line raster-454.
- [`analysis/notebooks/07_mint_lavender_ambient_classifier.ipynb`](analysis/notebooks/07_mint_lavender_ambient_classifier.ipynb)
  audits the balanced 45-trial ambient/mint/lavender panel, validates every
  prediction on a complete held-out trial, compares hierarchical and direct
  regularized models, reports class recall and uncertainty, audits the
  lavender/date confound, and freezes the closed-panel model only after its
  predeclared gate passes.
- [`analysis/protocols/mint-line-and-classifier-protocol.md`](analysis/protocols/mint-line-and-classifier-protocol.md)
  records the current experiment protocol.
- [`analysis/notebooks/08_mint_lavender_parallel_raster.ipynb`](analysis/notebooks/08_mint_lavender_parallel_raster.ipynb)
  applies the two frozen bundles to the parallel mint/lavender raster,
  registers the path to paper-local coordinates, and reports raw identity,
  temporal-context, uncertainty, support, and direct source-region diagnostics.
  Its primary comparison makes mint carryover visible by showing raw mint
  affinity beside the frozen active-mint result; source rectangles never alter
  the scores or maps.
- [`analysis/notebooks/09_long_random_sequence_shape_maps.ipynb`](analysis/notebooks/09_long_random_sequence_shape_maps.ipynb)
  applies the three frozen pilot bundles to the five reviewed July 30
  random-motion runs, verifies acquisition QC, maps current/active/fading
  evidence, and scores shape ranking against explicitly approximate source
  overlays. It finds the clearest spatial structure in the mint-only run and
  does not claim clean caret/V or lavender recovery.
- [`analysis/notebooks/10_long_sequence_qc_transfer_and_32d.ipynb`](analysis/notebooks/10_long_sequence_qc_transfer_and_32d.ipynb)
  audits all 11 reviewed long sequences, defines the first reproducible
  cleaning contract, visualizes the six August 3 trajectories, decomposes the
  frozen identity score into all 32 sensor contributions, and tests early
  mint-to-lavender versus lavender-to-mint transition patterns. Its source
  masks are protocol approximations and its repeated transitions are
  descriptive within-run events, not independent causal trials.
- [`analysis/notebooks/11_cleaning_schema_and_sequence_holdout.ipynb`](analysis/notebooks/11_cleaning_schema_and_sequence_holdout.ipynb)
  freezes `long-sequence-cleaning-v1`, preserves all 11,082 source rows in an
  auditable processed table, and trains a deliberately simple regularized
  causal classifier on ten complete recordings while holding out the complete
  August parallel-strip recording. Temporal context improves primary holdout
  macro recall from 0.537 to 0.663, but median three-class whole-recording
  recall is only 0.516; the saved artifact is therefore a reproducible pilot,
  not a validated final odor model.

## Interpreting the current background-referenced raster

`Common spatial support` means a grid cell is displayed only when both the
blank and mint scans have at least two retained readings within the 3 cm
smoothing radius. White cells are therefore missing shared coverage, not zero
odor. Yellow in the mint panel indicates a large 32-sensor RMS change from that
run's purge reference; it is not by itself a mint-specific classification. The
third panel is the direct mint-minus-blank comparison.

The Desk Y = 34 cm result is a one-dimensional row summary obtained by taking
the median across Desk X. It does not prove that the two-dimensional feature is
horizontal.

The appended diagnostics in notebook 04 support the following interpretation:

- **The acquisitions are not height-matched at the source:** the retry's median
  hover is closer to its target, but 21/147 mint readings were still raised
  above 3.5 cm at Desk Y 31.2–43.8 cm. The background has no exclusions.
- **Overall coverage is high but locally biased:** 712 of 780 grid cells have
  common support after smoothing, yet 14.3% of the in-bounds mint readings were
  excluded by height and none of the background readings were.
- **Source loading is uncontrolled:** 96.3% of shared cells are positive and
  the apparent band is 14 cm wide. The response is not obviously electronically
  clipped, but the unknown excessive dose plausibly created a broad plume and
  prolonged sensor response.
- **Smoothing changes the apparent direction:** hotspot angle ranges from
  -27 degrees at 1.5 cm to +4 degrees at 2.0 cm and +20 degrees at 3.0 cm.
  The near-horizontal 2 cm map is a sensitivity result, not a justified choice
  of the correct map.
- **Time is strongly confounded with position:** scan progress versus Desk X
  has Spearman rho about -0.99. Response versus progress is +0.83 for background
  and +0.45 for mint, indicating a time-dependent sensor-state, recovery, or drift
  effect in addition to any spatial odor signal.
- **Line direction is not stable to lag:** the hotspot angle spans roughly
  54 degrees over the 2–4 s sensitivity test, while its center remains within
  about 1.6 cm. The data support a reasonably stable affected location more
  strongly than a recovered horizontal orientation.

The visible gaps are directly explained by the height mismatch. Excessive,
unquantified source loading plausibly explains why the response is widespread
and wider than the paper strip. Scan-time confounding, physical-lag uncertainty,
and smoothing then make the apparent direction unstable. The retry is useful
detectability evidence, but not source-width or concentration validation.

All top-down plots in notebook 04 are displayed in the operator's orientation:
both spatial axes are reversed so the physical scan start appears at the
top-left. Axis labels remain the original Desk X and Desk Y coordinates in
centimeters; the data and calculations are not transformed.

## Likely next stage

The notebook 04 diagnosis includes the blotter retry. It identifies a strong
mint-source response, remaining localized height-filter gaps, excessive unknown
source loading, scan-time confounding, and physical-lag sensitivity. The retry
should not be used to score recovered source width or concentration.

1. Add a live height cue or non-contact height guide before asking the operator
   to repeat another spatial scan. Manual hovering alone has now produced
   localized high-snout gaps in two mint runs despite deliberate effort.
2. Before the next spatial run, record the source center and both endpoints in
   desk coordinates. Keep a photograph or video frame showing those marks so
   position, angle, and width can be scored independently.
3. Use a measured, modest oil volume that wets the strip without saturating or
   pooling. Record the volume or verified drop count and the exposure delay.
4. Break the current near-perfect time-versus-Desk-X confound. Use premarked
   perpendicular crossing lanes in a non-monotonic or randomized X order
   instead of progressing continuously from one side of the map to the other.
   Use the same recorded order for the matched blank and mint runs.
5. Include clean-air recovery segments between crossings and record paired
   forward/reverse crossings where practical. This allows response persistence
   and physical lag to be estimated inside the geometry-validation block.
6. Use a fresh removable backing, fixed desk reference, matched
   blank-first/mint-second conditions, and the same height and speed targets.
   Repeat the block on independent fresh backings. Treat trials, not
   individual readings, as independent units.
7. Freeze notebook 06's selected temporal feature set and regularization before
   collecting more spatial data. Validate it on a new randomized-order crossing
   block with marked source coordinates and recovery pauses; do not retune it
   from that raster. A matched non-mint line raster can then test spatial
   specificity before attempting multiple simultaneous odor sources.
