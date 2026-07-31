# Experiment session manifest

A manifest is a catalog of the recording folders. It prevents a timestamp from
being the only explanation of what a session contains. Update this table when a
new session is reviewed and moved under `experiments/`.

| Stage | Session | Trial ID | Analysis status | Intended use |
|---|---|---|---|---|
| Pose tracking | `20260717_164553` | Not recorded | Exploratory | AprilTag pose recording |
| Pose tracking | `20260717_171550` | Not recorded | Exploratory | Early Cyranose pose recording |
| Pose tracking | `20260717_171650` | Not recorded | Exploratory | Early Cyranose pose recording |
| Exploratory mapping | `20260720_202349` | Not recorded | Exploratory | Pre-protocol synchronized session |
| Exploratory mapping | `20260721_120649` | Not recorded | Exploratory | Initial spatial odor/trajectory map |
| Response lag | `20260722_133122` | `lag_fwd_01` | Excluded | Initial forward attempt before stabilized reference setup |
| Response lag | `20260722_133327` | `lag_rev_01` | Excluded | Initial reverse attempt before stabilized reference setup |
| Response lag | `20260722_134544` | `lag_blank_01` | Excluded | Blank attempt before the desk tag was fixed |
| Response lag | `20260722_134738` | `lag_fwd_02` | Excluded | Exploratory forward attempt with changing desk reference |
| Response lag | `20260722_135138` | `lag_rev_02` | Excluded | Exploratory reverse attempt with changing desk reference |
| Response lag | `20260722_135940` | `lag_blank_fixed_01` | **Accepted control** | Fixed-tag dry-paper-towel control |
| Response lag | `20260722_140431` | `lag_fwd_fixed_01` | Excluded from primary pair | Usable recording, but paired lateral lane did not match |
| Response lag | `20260722_140800` | `lag_rev_fixed_01` | Excluded from primary pair | Usable recording, but paired lateral lane did not match |
| Response lag | `20260722_142426` | `lag_rev_fixed_02` | **Accepted** | Pair 2 reverse |
| Response lag | `20260722_142717` | `lag_fwd_fixed_02` | **Accepted** | Pair 2 forward |
| Response lag | `20260722_143007` | `lag_fwd_fixed_03` | **Accepted** | Pair 3 forward |
| Response lag | `20260722_143225` | `lag_rev_fixed_03` | **Accepted** | Pair 3 reverse; camera repositioned while desk reference remained fixed |
| Spatial mapping / line raster | `20260722_150515` | `line_raster_mint_01` | **Usable pilot** | First controlled serpentine mint-line raster; analyze with 3.0 s lag correction |
| Spatial mapping / line raster | `20260722_153840` | `line_raster_blank_01` | **Excluded: contamination suspected** | Blank towel was placed over the previous mint location; retain only as a carryover diagnostic, not a clean control or classifier example |
| Spatial mapping / line raster | `20260722_161100` | `line_raster_blank_fresh_01` | **Usable pilot control** | Clean block 01 blank-first raster; no height-filter exclusions, but blank was held closer than mint |
| Spatial mapping / line raster | `20260722_163441` | `line_raster_mint_fresh_01` | **Usable pilot; not geometry validation** | Mint-second raster; 20/200 in-bounds readings were above the height band at Desk Y 30.5–36.0 cm, creating localized source-region gaps |
| Spatial mapping / line raster | `20260722_171702` | `line_raster_mint_blotter_retry_01` | **Usable detectability pilot; dose/height limited** | Blotter mint-source retry against `line_raster_blank_fresh_01`; exact dose unknown and reportedly excessive; 21/147 in-bounds readings exceeded 3.5 cm at Desk Y 31.2–43.8 cm |
| Spatial mapping / line raster | `20260723_163527` | `line_raster_mint_dry_strip_01` | **Diagnostic; not promoted** | Day-old 17 × 1.7 cm dry mint strip on fresh towel; excellent digital sync, but only 69/174 reviewed raster rows passed height QC and the raster mint score was effectively zero |
| Spatial mapping / line raster | `20260723_165457` | `line_raster_mint_moderate_strip_01` | **Horizontal line raster-454; promoted temporal-model pilot, not geometry validation** | Fresh 17 x 3 cm strip with exactly three drops and five-minute wait; 238/238 digital matches. With the documented 2.0 s working correction and fixed reported source, current-only source-region AUC is 0.672; the locked, presence-gated temporal active-mint pipeline raises it to 0.938 and reduces outside-source scores above 0.5 from 60.5% to 22.2% |
| Spatial mapping / superseded two-source pilot | `20260727_175116` | `t_raster_mint_lavender_01` | **Superseded diagnostic; preserved outside current narrative** | First T-shaped two-source attempt. Retained under `spatial-mapping/superseded-pilots/` for provenance; replaced as the active Notebook 08 experiment by the more evenly sampled parallel-strip raster. |
| Spatial mapping / parallel two-source raster | `20260727_184252` | `two_horizontal_raster_mint_lavender_01` | **Current pilot; partial mint result, lavender and fading transfer failed** | Two parallel physical strips with a clean gap. Digital sync is 296/296. Frozen temporal active-mint ranking is 0.963 AUC versus 0.705 currently. The new stationary-validated lavender temporal model reduces the false first-downward gap allocation but active-lavender source ranking is only 0.648 and the lavender strip remains mint-like. Direction-aware fading AUC is 0.296/0.290 for mint/lavender, so neither fading field is validated as post-source recovery. |
| Spatial mapping / long random sequence | `20260730_152653` | `2026-07-30_mint_only_horizontal_random_10min_run01` | **Excluded: aborted start** | Five-second initial start with 11 readings; preserved under the July 30 batch but not analyzed. |
| Spatial mapping / long random sequence | `20260730_152754` | `2026-07-30_mint_only_horizontal_random_10min_run01` | **Usable; clearest shape-ranking result** | Mint-only horizontal random-motion run. 706/706 digital matches; host duration 392.2 s. Notebook 09 reports RMS source-ranking AUC 0.858 and frozen active-mint AUC 0.639 against protocol-approximate geometry. |
| Spatial mapping / long random sequence | `20260730_153631` | `2026-07-30_caret_mint_left_lavender_right_random_10min_run01` | **Usable acquisition; partial/negative shape transfer** | Caret with mint left and lavender right. 737/737 digital matches; host duration 409.6 s. Identity-independent odor structure is modest, while the frozen identity/active maps do not cleanly recover both arms. |
| Spatial mapping / long random sequence | `20260730_154848` | `2026-07-30_caret_lavender_left_mint_right_random_10min_run01` | **Usable acquisition; partial/negative shape transfer** | Swapped caret. 759/759 digital matches; host duration 421.8 s. The active-mint score has the strongest mixed-layout source ranking in Notebook 09 (AUC 0.651), but lavender localization remains weak. |
| Spatial mapping / long random sequence | `20260730_161358` | `2026-07-30_lavender_only_horizontal_random_10min_run01` | **Usable acquisition; negative lavender localization result** | Lavender-only horizontal run. 829/829 digital matches; host duration 460.7 s. Frozen current and active lavender rankings are approximately chance (AUC 0.482/0.494). |
| Spatial mapping / long random sequence | `20260730_163355` | `2026-07-30_inverted_caret_lavender_left_mint_right_random_10min_run01` | **Usable 10-minute acquisition; partial/negative shape transfer** | Latest inverted-caret V run. 1,114/1,114 digital matches; host duration 619.3 s. Overall RMS ranks the union of the approximate source arms at AUC 0.700, but mint/lavender identity and active-state maps do not recover the two arms cleanly. |

| Classifier | `mint-identity-pilot-01` | 70 stationary PCnose+ trials | **Accepted pilot dataset** | 15 blank, 15 mint, 20 bergamot, and 20 lemongrass trials; unmodified repository copies indexed by `trial_manifest.csv` |
| Classifier | `mint-lavender-ambient-pilot-01` | 45 stationary PCnose+ trials | **Accepted closed-panel pilot; session-limited** | 15 ambient, 15 mint, and 15 lavender trials. Complete-trial hierarchical validation achieved 15/15 recall for each class; direct three-class balanced accuracy was 86.7%. Lavender was recorded on a later date, so session-invariant and mixture recognition remain untested |

Current stationary mint/lavender/ambient analysis:

- all 45 repository copies match their recorded source SHA-256 digest, contain
  32 sensor channels, and contain 58 flag-3 exposure rows;
- three nonfatal phase-count deviations are retained as explicit manifest
  warnings: `blank-3` and `lavender-9` each have one extra flag-1 row, while
  `blank-10` has one fewer flag-7 row;
- notebook 07 fixes regularization at the value selected before lavender was
  inspected, then performs five-fold validation with complete trials held out;
- the primary hierarchy separates ambient-versus-odor from
  direction-normalized mint-versus-lavender identity. It achieved 100.0%
  balanced accuracy and 15/15 recall for every class. With 15 trials per class,
  each perfect recall has an approximate 79.6-100.0% Wilson interval;
- the direct three-class diagnostic achieved 86.7% balanced accuracy:
  ambient 15/15, mint 12/15, and lavender 12/15;
- all lavender trials were collected on July 27, while mint and ambient were
  collected July 22-23. The closed-panel result therefore cannot distinguish
  lavender identity from recording-session drift;
- the accepted model is frozen under
  `classifier/mint-lavender-ambient-pilot-01/frozen-models/mint-lavender-ambient-v1/`
  with its model card, configuration, manifest digest, checksum, and exact
  reload verification;
- notebook 08 adds `temporal-lavender-seeker-v1`, trained without raster
  outcomes. All 15 lavender files have valid exposure/recovery sequences.
  Complete-trial causal validation reaches 81.2% balanced accuracy with
  active/fading AUC 0.968/0.929; the current-only snapshot is slightly better
  at 84.0%. The frozen SHA-256 is
  `c11d837492b30c2d0c824826c3819749756a73a9698c1992f54ef9d7ff65228a`;
- this model has not been trained or validated on mixtures. Notebook 08 now
  supplies the current exploratory parallel-strip moving-raster application,
  whose partial/negative transfer result is documented below rather than
  treated as model validation.

Current two-source parallel-strip analysis:

- the reviewed session is stored under
  `spatial-mapping/parallel-strips/mint-lavender-parallel-pilot-01/`;
- notebook 08 verifies and applies `mint-lavender-ambient-v1`,
  `temporal-mint-seeker-v1`, and `temporal-lavender-seeker-v1`;
- its paper-local 26.5 x 26.5 cm registration uses mint X 5-22, Y 6-8 cm
  and lavender X 5-22, Y 18.5-21 cm. The operator's top/bottom measurements
  were transposed into conventional plot X/Y so the overlays match the
  physically horizontal strips;
- Paper Y is `46.5 - desk_y`, fixed from the confirmed bottom-left start,
  first upward pass, and second downward pass without inspecting odor scores.
  The stationary pre-raster dwell is excluded at the geometry-confirmed
  40.0 s raster start;
- digital synchronization is excellent (296/296 matches; 96.3 ms p95);
- the 2.0 s paper-local view retains 232 causal-history readings, of which 114
  pass strict 1.0-3.5 cm height QC and 230 enter the explicitly labeled
  soft-height support calculation;
- strict direct sampling reaches both source rectangles: 5 mint rows and
  9 lavender rows, with 13 and 17 readings within 1 cm, respectively;
- raw mint source-ranking is AUC 0.705, while the unchanged temporal
  active-mint pipeline raises it to 0.963 and reduces strict background
  readings above 0.5 from 69.0% to 28.0%;
- lavender current-identity ranking is AUC 0.776, while active-lavender
  ranking is 0.648 and the reported lavender strip remains predominantly
  mint-like;
- on the raw first downward lane, lavender allocation is already median 0.535
  and maximum 0.896 in the gap before physical lavender exposure. Temporal
  active-lavender evidence reduces the gap median to 0.193, while fading
  lavender is 0.006. This identifies the original purple gap as classifier
  domain transfer rather than verified lavender transport;
- direction-aware fading AUC is 0.296 for mint and 0.290 for lavender. Neither
  fading map ranks post-source readings above pre-source readings;
- notebook 08 replaces the categorical temporal-context label panel with
  continuous closed-panel allocations and symmetric active/fading maps. None
  represents concentration, calibrated mixture proportions, or zero-shot
  decomposition.

Current long random-sequence analysis:

- the five promoted July 30 sessions are stored together under
  `spatial-mapping/random-waypoint-sequences/2026-07-30-long-sequence-pilot-01/`;
- every promoted run has 100% pose/readout matching within 250 ms, complete
  32-channel readings, and usable paper-local pose coverage;
- notebook 09 hash-verifies and applies the frozen identity, mint-temporal, and
  lavender-temporal bundles without retraining or selecting parameters from the
  spatial outcomes;
- temporal histories use host-derived timestamps. The original guide's
  device-time timer advanced about 1.6 times faster than wall time and shortened
  the first four runs, but did not speed up or corrupt the recorder streams;
- the fixed 2.0 s correction, 1.0-3.5 cm height band, paper registration, and
  gridding policy are shared across all five runs;
- source overlays are protocol-approximate because exact strip vertices were
  not measured. They are never used to train or tune the models;
- the mint-only horizontal run preserves the strongest direct spatial ranking
  (RMS AUC 0.858; active-mint AUC 0.639);
- across odor/source comparisons, median current identity AUC is 0.523 and
  median active-state AUC is 0.524, while median identity-independent any-odor
  AUC is 0.646. Median active-map IoU at the working 0.5 threshold is zero;
- the latest inverted-caret run is a healthy acquisition with modest
  identity-independent spatial structure (RMS AUC 0.700), but neither the V
  geometry nor its mint/lavender assignment is cleanly recovered. This is a
  model-transfer/physical-response limitation, not a synchronization failure.

Primary response-lag dataset:

- control: `lag_blank_fixed_01`;
- Pair 2: `lag_rev_fixed_02` + `lag_fwd_fixed_02`;
- Pair 3: `lag_fwd_fixed_03` + `lag_rev_fixed_03`.

Current line-raster analysis:

- notebook 04 now uses `line_raster_mint_blotter_retry_01`, stored under
  `spatial-mapping/line-raster/background-referenced/blotter-retry-01/`,
  against the earlier clean background `line_raster_blank_fresh_01`;
- the retry produces a strong widespread response (3.99 times the background
  median RMS and 96.3% positive shared grid cells), but the exact excessive
  source dose is unknown and the apparent band is about 14 cm wide;
- manual hover improved overall but still created 21 height-filter exclusions
  around the source region. Preserve this as a detectability pilot, not source
  width or concentration validation;
- the day-old dry-strip trial is stored under
  `spatial-mapping/line-raster/source-loading-diagnostics/dry-strip-01/`;
- its 284/286 digital matches are sound, but only 39.7% of reviewed raster
  readings meet the current height band. Its raster classifier peak is
  effectively zero and its approximate source-region AUC is 0.490, so notebook
  05 preserves it as an under-responsive source diagnostic rather than
  replacing the earlier matched line panel;
- the measured moderate-source trial is displayed as **Horizontal line
  raster-454** and stored under
  `spatial-mapping/line-raster/horizontal-line-raster-454/`;
- its 238/238 digital matches are sound and the measured three-drop source
  restores a strong response. The approximate source region ranks above the
  outside region (AUC 0.835), but outside scores remain elevated, the score-time
  correlation is +0.51, and only 61.1% of reviewed readings pass the height
  band. Notebook 05 promotes it as a source-loading/detectability pilot, not
  a recovered-width result;
- notebook 06 locks a three-state causal temporal model using only the 70
  stationary PCnose+ trials, then applies it to Horizontal line raster-454.
  Held-out-trial balanced accuracy improves from 83.9% for current-only
  snapshots to 87.7% with temporal history. On the raster, active-mint
  source-ranking AUC improves from 0.672 to 0.938, outside-source readings above
  0.5 fall from 60.5% to 22.2%, and the deployed score-time correlation changes
  from +0.53 to +0.28 after the required odor-presence gate, using the 2.0 s
  working correction and fixed reported source coordinates. These are pilot
  localization results, not calibrated probabilities or recovered-width
  validation;
- notebook 06 also performs a fixed-ground-truth lag audit using the same 157
  reviewed readings at 0.25-second increments from 0 to 5 seconds. Before
  gridding or smoothing, active-score-weighted distance outside the reported
  strip is minimized at 2.25 s; distance and AUC jointly support 2.25-2.50 s,
  versus the prior 3.0 s assumption. Upward and downward pass minima are 2.50 s
  and 2.25 s. The reported strip coordinates are approximate, so this is a
  provisional calibration range that must be frozen and checked on a new run,
  not an independent spatial-validation result;
- Notebook 06 uses 2.0 s as its documented working spatial correction: a
  directly tested conservative compromise between 1.5 and 2.25 s that reduces
  the 3.0 s overcorrection while avoiding false precision. Its before/after
  spatial views and metrics are rerun consistently at 2.0 s, and the reported
  strip coordinates remain fixed independently of lag;
- a final lag-consensus ceiling check takes the supported-cell median across
  1.50, 1.75, 2.00, and 2.25 s, then reports a separate agreement-weighted
  view. It uses no source mask, clipping, or geometry prior. Agreement weighting
  reduces the thresholded outside-strip fraction from 70.8% to 66.7% and the
  above-strip extension from 7.3 to 6.3 cm, but the residual field remains
  irregular. Lag selection alone is therefore not treated as capable of
  recovering the reported rectangle from this run;
- for the notebook 06 raster window, all 61 strict height exclusions were
  below 1.0 cm and none were above 3.5 cm; 52 were only 0.75-1.0 cm. The strict
  heatmap retains 63.2% grid support. A labeled secondary presentation map
  softly weights readings from 0.5-1.0 cm, keeps the 3 cm spatial radius and
  two-reading minimum, and raises displayed support to 88.5%. It does not
  replace the strict QC result;
- the locked presence, mint-identity, and temporal models are frozen together
  as `classifier/mint-identity-pilot-01/frozen-models/temporal-mint-seeker-v1`.
  The bundle records its 70-trial training-data digest, sensor order, temporal
  feature configuration, regularization, baseline envelope, model checksum,
  and limitations. Later spatial shapes must load it unchanged;
- the accompanying deployment policy requires odor-presence evidence of at
  least 0.5 before temporal active-mint evidence can enter a spatial map.
  Recovery remains ungated. On the known-clean first lane, odor presence had
  median 0.047 and 0/34 readings passed the gate, so all startup active-state
  false positives are removed without changing the frozen model. The gate
  changes the after-temporal tracked raster, heatmap, and active-mint
  diagnostics only; before-temporal plots and recovery evidence remain
  unchanged. The dashed 0.5 contour on the after-temporal heatmap separates
  working detections from faint sub-threshold interpolation;
- the clean-block-01 pilot pair is stored under
  `spatial-mapping/line-raster/matched-pairs/clean-block-01/`;
- `line_raster_blank_fresh_01` was recorded first and
  `line_raster_mint_fresh_01` second without moving the desk tag, camera,
  raster bounds, or lane guides;
- notebook 04 performs a shared-support, 3.0 s lag-corrected direct map
  subtraction and audits the diagonal feature. The mint scan was systematically
  raised above the accepted height at the apparent line crossing, so this pair
  supports a condition difference but not line-geometry validation;
- the physical source center and endpoints were not independently recorded,
  so spatial accuracy relative to the true line cannot yet be scored.

Earlier line-raster pilot:

- `line_raster_mint_01` is preserved under
  `spatial-mapping/line-raster/pilot-usable/`;
- it is evidence of spatial signal concentration, not yet a definitive line
  reconstruction;
- its first intended blank control was contaminated by setup and is excluded
  below.

Contamination-suspected control attempt:

- `line_raster_blank_01` is preserved under
  `spatial-mapping/line-raster/excluded-controls/contamination-suspected/`;
- the blank towel was placed over the desk area previously occupied by the
  mint towel, so its near-line signal may represent residual surface odor;
- do not use this run as a clean control or classifier-training example;
- it has been superseded for pilot comparison by the reviewed clean-block-01
  blank-first/mint-second pair, which is itself not valid for geometry scoring.
