# Mint, lavender, and ambient pilot 01

This experiment is the first closed-panel stationary test of whether the
Cyranose 320 can distinguish:

- ambient room air;
- mint headspace;
- lavender headspace.

The dataset contains unmodified repository copies of 45 PCnose+ recordings:

- 15 existing ambient trials;
- 15 existing mint trials;
- 15 lavender trials recorded on July 27, 2026.

Original files remain in their source locations. The copied files under
`raw/pcnose/` must not be rewritten. `trial_manifest.csv` records the source
path, SHA-256 digest, device metadata, phase counts, and quality-control
status for every trial.

The primary analysis is
`analysis/notebooks/07_mint_lavender_ambient_classifier.ipynb`.
It uses each trial's late flag-1 readings as its reference, converts flag-3
sample readings to percentage change from that reference, and validates
predictions by holding out complete trials. Scaling and model fitting occur
inside each training fold.

The primary model is hierarchical:

1. ambient versus any recorded odor;
2. mint versus lavender using direction-normalized 32-sensor fingerprints.

A direct three-class regularized logistic model is retained as a diagnostic
comparison. This pilot does not train on odor mixtures and does not claim
mixture decomposition.

Important limitation: lavender was recorded on a later date than the mint and
ambient trials. Complete-trial validation prevents row leakage, but it cannot
fully separate odor identity from recording-session drift. Successful results
therefore establish separability within this recorded dataset, not universal
or session-invariant odor recognition.

## Quality-control result

- All 45 repository copies match their source SHA-256 digest.
- Every trial has all 32 sensors and 58 flag-3 exposure rows.
- `blank-3` and `lavender-9` have 21 rather than 20 flag-1 rows.
- `blank-10` has 57 rather than 58 flag-7 rows.
- Those three files remain usable because their reference and active-exposure
  data are complete; the deviations are recorded as manifest warnings.
- No duplicate raw files or nonfinite flag-3 sensor values were found.

## Closed-panel validation result

The regularization setting (`C=0.1`) was fixed from the earlier classifier
pilot before lavender performance was inspected.

Five-fold validation held out three complete trials from every class in each
fold. The primary hierarchy achieved:

- 100.0% accuracy and balanced accuracy;
- macro F1 1.000;
- ambient recall 15/15;
- mint recall 15/15;
- lavender recall 15/15.

With only 15 trials per class, a 15/15 recall still has a 95% Wilson interval
of approximately 79.6-100.0%. The direct three-class diagnostic achieved 86.7%
balanced accuracy: ambient recall was 15/15, while mint and lavender were each
12/15. This contrast supports the hierarchy's deliberate separation of odor
presence from direction-normalized identity, but it does not remove the
recording-session limitation.

The accepted closed-panel bundle is frozen under
`frozen-models/mint-lavender-ambient-v1/`. It reloads with a maximum absolute
score difference of zero. The bundle is not a mixture model and has not been
tested on moving spatial scans.

## Lavender exposure/recovery model

Notebook 08 reuses the same 45 stationary files to train a three-state causal
pilot:

- active lavender;
- fading lavender during flags 6 and 7;
- no active lavender, including ambient and mint controls.

All 15 lavender files contain the expected `1 -> 3 -> 6 -> 7` phase sequence.
Nested validation holds out complete files and selects the temporal feature
configuration without using the spatial raster. The locked six-second
difference model (`C=0.3`) achieved:

- 81.2% balanced state accuracy;
- 0.968 active-lavender AUC;
- 0.929 fading-lavender AUC;
- 87.0% active recall;
- 70.9% fading recall;
- 85.7% no-active recall.

The current-only snapshot reached 84.0% balanced accuracy, so temporal history
adds causal state interpretation but does not improve every stationary
metric. The pilot passed its written adequacy gate and is frozen under
`frozen-models/temporal-lavender-seeker-v1/` with SHA-256
`c11d837492b30c2d0c824826c3819749756a73a9698c1992f54ef9d7ff65228a`.
Its model card records the date/session limitation, stationary-to-moving
transfer risk, and absence of mixture validation.
