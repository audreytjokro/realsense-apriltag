# Valid lag-test sessions

These sessions are the primary dataset for estimating the Cyranose's physical
odor-response lag. The original CSV, video, metadata, and alignment-summary
files have not been modified.

Recommended analysis settings:

- Digital Cyranose/pose matching threshold: 250 ms
- Physical odor-response lag: 3.0 s
- Lag sensitivity range: 2.5-3.5 s
- Use host/pose time for physical seconds; do not use `pcnose_device_time_s`

Sessions:

- `cyranose_reading_pose_session_20260722_135940`: fixed-desk-tag blank control (`lag_blank_fixed_01`)
- `cyranose_reading_pose_session_20260722_142426`: Pair 2 reverse (`lag_rev_fixed_02`)
- `cyranose_reading_pose_session_20260722_142717`: Pair 2 forward (`lag_fwd_fixed_02`)
- `cyranose_reading_pose_session_20260722_143007`: Pair 3 forward (`lag_fwd_fixed_03`)
- `cyranose_reading_pose_session_20260722_143225`: Pair 3 reverse (`lag_rev_fixed_03`)

Pairs 2 and 3 are the primary lag pairs because their opposing trajectories
substantially overlap in desk coordinates. They produced physical-lag
estimates of 2.90 s and 2.79 s, respectively.
