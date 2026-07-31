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

- [`2026-07-30-long-sequence-pilot-01/`](2026-07-30-long-sequence-pilot-01/)
  contains the five usable mint/lavender horizontal, caret, and inverted-caret
  sequences plus one preserved aborted start. Notebook 09 performs the current
  shape-preservation audit.

The waypoint guide now derives elapsed time from the recorder's host midpoint
timestamp. In the first July 30 batch, the earlier guide used the Cyranose
device counter, which advanced about 1.6 times faster than wall clock. This
shortened the first four operator-timed runs but did not alter or invalidate
their synchronized raw data. For those older waypoint logs,
`host_timestamp_utc` is valid and `recorder_elapsed_s` is not physical time.
