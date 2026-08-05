# Random-waypoint moving sequences

This stage is reserved for longer synchronized Cyranose + RealSense recordings
collected with approximately random motion over the 26.5 x 26.5 cm paper-local
sampling area.

The waypoint guide is optional. It provides reproducible suggestions and a
recording timer but does not control the operator, Cyranose, RealSense, or
recorder. Each completed timestamped recorder session should retain:

- `cyranose_reading_pose.csv`;
- `rectified_rgb.mp4`;
- `alignment_summary.json`;
- `session_metadata.json`;
- `waypoint_sequence.csv`;
- `waypoint_metadata.json`.

Keep each physical layout fixed for its entire recording. Preserve source
dimensions, paper-local positions, strip assignments, dose, preparation time,
room conditions, and a setup photograph with the reviewed session.

Reviewed batches:

- [`2026-07-30_batch-01/`](2026-07-30_batch-01/)
  contains the five usable mint/lavender horizontal, caret, and inverted-caret
  sequences plus one preserved aborted start. Notebook 09 performs the current
  shape-preservation audit.
- [`2026-08-03_batch-01/`](2026-08-03_batch-01/)
  contains the six reviewed X-shape, inverted-caret, and parallel-strip
  long-sequence recordings collected on August 3.

Cross-batch analysis is in
[`../../analysis/notebooks/10_long_sequence_qc_transfer_and_32d.ipynb`](../../analysis/notebooks/10_long_sequence_qc_transfer_and_32d.ipynb).
It audits all 11 reviewed runs without rewriting raw data, defines the proposed
first cleaning contract, visualizes the six August trajectories, preserves all
32 sensor dimensions in the identity-contribution view, and reports an
exploratory directional carryover pattern with explicit geometry and
independence limitations.

The versioned processed export and pilot sequence model are under
[`processed/cleaning-v1/`](processed/cleaning-v1/). Notebook 11 preserves all
11,082 raw rows with explicit lineage and exclusion reasons, then holds out one
complete recording for evaluation. This is the first frozen preprocessing
contract; the accompanying classifier remains a small-data pilot.

The waypoint guide now derives elapsed time from the recorder's host midpoint
timestamp. In the first July 30 batch, the earlier guide used the Cyranose
device counter, which advanced about 1.6 times faster than wall clock. This
shortened the first four operator-timed runs but did not alter or invalidate
their synchronized raw data. For those older waypoint logs,
`host_timestamp_utc` is valid and `recorder_elapsed_s` is not physical time.
