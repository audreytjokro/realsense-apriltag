# 2026-07-30 long random-waypoint sequence pilot

This reviewed batch contains five synchronized Cyranose + RealSense moving
sequences over a 26.5 x 26.5 cm paper-local area. Motion was intentionally
approximately random instead of a sequential raster. All five promoted runs
have 100% pose/readout matching within 250 ms and complete 32-channel sensor
vectors.

## Included runs

| Layout | Session | Host duration | Readings | Sync p95 | Status |
|---|---|---:|---:|---:|---|
| Mint-only horizontal strip | `20260730_152754` | 392.2 s | 706 | 85.3 ms | Usable |
| Caret: mint left, lavender right | `20260730_153631` | 409.6 s | 737 | 95.2 ms | Usable |
| Caret: lavender left, mint right | `20260730_154848` | 421.8 s | 759 | 92.9 ms | Usable |
| Lavender-only horizontal strip | `20260730_161358` | 460.7 s | 829 | 88.8 ms | Usable |
| Inverted caret: lavender left, mint right | `20260730_163355` | 619.3 s | 1,114 | 89.7 ms | Usable |

The five-second `20260730_152653` mint-only start is preserved under
`excluded-aborted/` and is not analyzed.

## Timing note

The first four recordings are shorter than the intended ten minutes because
the original waypoint guide treated the Cyranose device counter as wall-clock
seconds. That counter advanced about 1.6 times faster than real time. The raw
recorder data are not sped up or corrupted: their host timestamps, RealSense
timestamps, video, and alignment remain valid. The guide now calculates its
elapsed display from `pcnose_sample_time_estimate_utc`. The final inverted-caret
run used host timing and lasted 619.3 seconds.

The older waypoint CSV files retain correct `host_timestamp_utc` values, but
their recorded `recorder_elapsed_s` field reflects the old fast device counter
and must not be interpreted as physical seconds.

## Shape analysis

[`09_long_random_sequence_shape_maps.ipynb`](../../../../analysis/notebooks/09_long_random_sequence_shape_maps.ipynb)
loads the three frozen pilot bundles without retraining and applies one fixed
analysis policy to all five runs:

- host-derived time for causal temporal features;
- a fixed 2.0 s physical response correction;
- the existing 1.0-3.5 cm strict height band;
- the existing paper registration;
- protocol-approximate source overlays used only for scoring and display.

The source endpoints were not measured in desk coordinates. The notebook
therefore uses clearly labeled, protocol-approximate 17 x 3 cm overlays and
does not treat them as exact ground truth.

Main result: the mint-only horizontal run retains the clearest spatial
ranking. Its identity-independent RMS field has source-versus-background AUC
0.858, and the frozen active-mint score has AUC 0.639. Across the three
two-source layouts, identity-independent odor structure is modest, while the
mint/lavender identity and active-state maps do not cleanly recover the caret
or inverted-caret geometry. The lavender-only run is approximately chance for
lavender localization. These are usable recordings with a negative/partial
model-transfer result, not corrupted acquisitions.

See `trial_manifest.csv` for the compact batch index.
