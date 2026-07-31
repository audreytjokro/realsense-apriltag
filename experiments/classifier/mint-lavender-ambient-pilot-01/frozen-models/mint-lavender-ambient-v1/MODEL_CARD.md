# Mint, lavender, and ambient v1

Status: accepted closed-panel stationary pilot.

Training data:

- 15 ambient trials;
- 15 mint trials;
- 15 lavender trials;
- 32 unfiltered Cyranose 320 channels;
- complete trials held out during five-fold validation.

Primary held-out-trial results:

- accuracy: 100.0%;
- balanced accuracy: 100.0%;
- macro F1: 1.000;
- per-class recall:
  - ambient: 100.0%;
  - mint: 100.0%;
  - lavender: 100.0%;

The model is hierarchical: ambient-versus-odor followed by direction-normalized
mint-versus-lavender identity. Regularization C=0.1 was fixed from the
earlier classifier pilot before lavender performance was inspected.

Limitations:

- lavender was recorded on a later date than the existing mint and ambient
  trials, so lavender identity is confounded with recording session;
- this is a stationary closed-panel result;
- no odor mixture was used for training or validation;
- scores are affinities, not concentrations or universal calibrated
  probabilities;
- spatial transfer has not been tested.

Model SHA-256: `73d292015272a6fe69ee0bd2ce35f8fd85769068eb6b213b9acaa5ebb5e08b83`
Manifest SHA-256: `102ea67a5d5e3777b8e131c044f4d1cac98f7457da2268222f7c6eb289cbc551`
Reload maximum absolute difference: `0.000e+00`
