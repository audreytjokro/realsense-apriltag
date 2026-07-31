# Temporal lavender seeker v1

Status: frozen causal-state pilot; moving-scan validation failed.

This artifact separates:

- active lavender;
- fading lavender during PCnose+ recovery phases;
- no active lavender.

Training uses the 45 stationary ambient, mint, and lavender files. Complete
trials are held out during nested validation; spatial raster positions and
scores are not used to choose features or regularization.

Held-out stationary results:

- causal balanced accuracy: 81.2%;
- active-lavender AUC: 0.968;
- fading-lavender AUC: 0.929;
- active recall: 87.0%;
- fading recall: 70.9%;
- no-active recall: 85.7%.

The current-only snapshot reaches 84.0% balanced accuracy, so temporal history
adds causal semantics but does not improve every stationary metric.

Locked configuration:

- temporal differences at 0.5, 1.5, 3.0, and 6.0 seconds;
- no EMA trace features;
- regularization `C=0.3`;
- minimum history 6.0 seconds;
- active evidence uses the frozen odor-presence gate and lavender identity
  direction;
- fading evidence remains ungated because recovery can persist as current odor
  presence falls.

Model SHA-256:
`c11d837492b30c2d0c824826c3819749756a73a9698c1992f54ef9d7ff65228a`

Limitations:

- only 15 independent lavender exposure/recovery trials;
- lavender was recorded in a later session than mint and ambient;
- stationary PCnose+ purge phases differ from open-air motion;
- no mixture training or validation;
- scores are evidence values, not calibrated concentrations;
- the current parallel-strip raster does not validate lavender localization or
  directionally correct fading transfer.
