# Analysis

The numbered notebooks are the primary analysis record. Each notebook contains
its data selection, quality-control rules, calculations, visualizations, and a
short interpretation. They are saved with executed outputs.

1. `notebooks/01_trajectory_odor_maps.ipynb`: exploratory trajectory and odor
   maps.
2. `notebooks/02_response_lag_validation.ipynb`: accepted forward/reverse lag
   pairs and the 3.0 s working correction.
3. `notebooks/03_line_raster_mint_01.ipynb`: the first controlled mint-line
   raster pilot.
4. `notebooks/04_mint_vs_blank_raster.ipynb`: current blotter mint-source versus
   clean-background analysis with shared-support subtraction, raw and height
   audits, scan-time diagnostics, smoothing/lag sensitivity, spatial profiles,
   and median 32-sensor fingerprints.
5. `notebooks/05_mint_identity_classifier.ipynb`: hybrid row/window classifier
   with trial-grouped validation, blank-versus-odor detection, mint identity
   against bergamot and lemongrass, a regularized logistic-versus-MLP check,
   application to the current blotter raster, and exploratory transfer checks
   on the older rectangle and paper-towel line scans. It also compares the
   day-old dry strip with the measured three-drop moderate strip to separate
   source-loading failure from remaining persistence and height limitations.
6. `notebooks/06_dynamic_mint_exposure_recovery.ipynb`: three-state
   active-mint, mint-recovery, and no-active-mint model. It compares
   current-only and causal temporal features with nested validation that keeps
   complete stationary trials together, locks the selected model without using
   spatial outcomes, and applies it to Horizontal line raster-454. The final
   presentation section contains visually matched before/after tracked-raster
   charts, a softly height-weighted before-temporal heatmap, and an
   agreement-weighted lag-consensus after-temporal heatmap. The temporal chart
   keeps active mint, recovery, and snout height together on two labeled axes.
   Active-mint display and mapping apply the frozen odor-presence gate before
   smoothing, while recovery remains ungated. The notebook also freezes and
   reload-verifies `temporal-mint-seeker-v1` and its deployment policy. The
   after-temporal heatmap includes a dashed 0.5 working-detection contour so
   faint sub-threshold interpolation is not mistaken for detected mint. A
   lag-sensitivity audit remaps the same 157 readings over 0-5 seconds in
   0.25-second steps, holds the reported strip fixed, evaluates raw unsmoothed
   distance and source-ranking metrics, and shows matched heatmaps at selected
   lags. It supports a provisional 2.25-2.50 s calibration band without
   changing the frozen model. The notebook uses 2.0 s for its presentation and
   dependent spatial diagnostics as a conservative, directly tested compromise
   between 1.5 and 2.25 s; it holds the reported source coordinates fixed so
   changing lag cannot move the target. Its final ceiling check then takes the
   median across 1.50, 1.75, 2.00, and 2.25 s and separately weights that
   consensus by cross-lag detection agreement. This modestly reduces the
   off-strip field but does not recover the rectangular source, showing that
   timing choice alone cannot remove the remaining persistence and plume shape.
   Its presentation output displays only this agreement-weighted consensus
   heatmap, while retaining the component calculations for auditability.
7. `notebooks/07_mint_lavender_ambient_classifier.ipynb`: balanced
   45-trial stationary classifier audit for ambient, mint, and lavender.
   Five-fold validation holds out complete trials, compares a hierarchical
   odor-presence plus direction-normalized identity model against a direct
   three-class logistic diagnostic, reports confusion matrices and per-class
   recall, audits recording-date confounding, and conditionally freezes the
   accepted `mint-lavender-ambient-v1` bundle. The hierarchy achieved 15/15
   recall for every recorded class; the direct model achieved 86.7% balanced
   accuracy. Because all lavender trials came from a later recording date,
   this is a closed-panel result rather than proof of session-invariant or
   mixture recognition.
8. `notebooks/08_mint_lavender_parallel_raster.ipynb`: current moving
   two-source transfer pilot on a 26.5 x 26.5 cm paper-local surface with two
   parallel strips and a clean gap. It verifies and loads
   `mint-lavender-ambient-v1` and `temporal-mint-seeker-v1` unchanged. It also
   complete-trial-validates and freezes `temporal-lavender-seeker-v1` from the
   stationary files only (81.2% causal balanced accuracy; active/fading AUC
   0.968/0.929), while reporting that the current-only snapshot is slightly
   better at 84.0%. The spatial section applies the documented causal features,
   2.0 s working pose correction, soft height weighting, and 1.50-2.25 s
   lag-consensus mapping. It shows continuous current allocations, symmetric
   active/fading mint and lavender heatmaps, and one combined active-evidence
   map: exclusive mint is green, exclusive lavender is purple, and their
   shared/ambiguous component is red. This combined view preserves continuous
   sub-threshold evidence. The four-panel diagnostic uses a labeled 0-0.5
   continuous scale for active lavender (observed maximum 0.441) and the
   observed 0-0.1167 range for fading lavender, so weak signals remain visible
   without presenting them as mint-scale evidence. It also includes an
   unsmoothed first-downward-lane audit and direction-aware fading diagnostics.
   Active mint ranks its strip
   at AUC 0.963 versus 0.705 currently; active lavender ranks at 0.648 versus
   0.776 currently and is essentially absent over the reported lavender strip.
   Raw lavender allocation is already high in the first downward gap before
   lavender exposure (median 0.535; maximum 0.896), while the temporal active
   score reduces it to 0.193. Direction-aware fading AUC is 0.296/0.290 for
   mint/lavender, so neither fading map is validated as post-source recovery.
   Paper Y is reflected from the confirmed scan order and pre-raster dwell is
   excluded at 40.0 s. This remains a partial mint result and a negative
   lavender/fading moving-transfer result, not two-odor decomposition.
9. `notebooks/09_long_random_sequence_shape_maps.ipynb`: frozen-model
   shape-preservation audit of five July 30 approximately random-motion
   sequences: mint-only, lavender-only, two caret assignments, and one inverted
   caret. It uses host-derived elapsed time for causal histories, the fixed
   2.0 s pose correction, the existing height/support rules, and
   protocol-approximate source overlays that never alter model scores. All five
   acquisitions pass synchronization and sensor-integrity checks. The
   mint-only run retains the clearest ranking (RMS AUC 0.858; active-mint AUC
   0.639). Median current-identity and active-state AUC across the source
   comparisons is only 0.523/0.524, versus 0.646 for identity-independent odor
   evidence. The caret/V shapes are therefore not cleanly reconstructed,
   especially for lavender; the result is a useful partial/negative moving
   transfer audit rather than evidence of corrupted data.
10. `notebooks/10_long_sequence_qc_transfer_and_32d.ipynb`: cross-batch QC and
   exploratory sequence audit for all 11 reviewed July 30 and August 3 runs.
   It applies one read-only cleaning policy, displays the six newest
   trajectories with green/yellow/red time progression, and shows every
   sensor's exact contribution to the frozen mint-versus-lavender identity
   logit. All 11 runs pass the stated readiness gate. The directional
   transition result is strongly mint-dominant but remains descriptive because
   source geometry is approximate and repeated events within a run are not
   independent.
11. `notebooks/11_cleaning_schema_and_sequence_holdout.ipynb`: freezes the
   first long-sequence cleaning contract and exports every raw row with its
   32 raw, normalized, and causal-filtered channels plus pose, geometry,
   eligibility, and exclusion provenance. It predeclares `20260803_170357` as
   the complete primary holdout, trains on the other ten recordings, and uses
   fixed-spec leave-one-recording-out auditing. Temporal features improve the
   primary macro recall from 0.537 to 0.663, while the median across nine
   three-class held-out runs is 0.516. This supports continued pilot work, not
   a claim of generalizable chemical identification.

Supporting folders:

- `reports/`: optional HTML exports of the notebooks for browser-only viewing.
- `figures/`: diagnostic frames and contact sheets used during review.
- `protocols/`: experimental procedures and design notes.
- `manifests/`: templates and analysis-side indexes.

To rerun a notebook from the repository root:

```powershell
$env:PYTHONNOUSERSITE = "1"
jupyter lab analysis/notebooks/07_mint_lavender_ambient_classifier.ipynb
```
