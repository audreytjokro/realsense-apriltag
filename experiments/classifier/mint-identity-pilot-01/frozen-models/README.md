# Frozen mint-seeker models

Each versioned subfolder contains a fitted model bundle, a human-readable model
card, and a SHA-256 checksum. A frozen version must not be retrained or tuned
using later spatial shapes.

`temporal-mint-seeker-v1` is selected and trained only from the 70 stationary
PCnose+ trials in `mint-identity-pilot-01`:

- 15 blank;
- 15 mint;
- 20 bergamot;
- 20 lemongrass.

The bundle contains the original odor-presence and mint-identity models plus the
three-state temporal model for active mint, mint recovery, and no active mint.
It also records the sensor order, selected causal feature configuration,
regularization, baseline envelope, training-data digest, and library versions.

Notebook 06 creates the artifact once, reloads it, and verifies that the frozen
temporal model reproduces the in-memory raster scores. Future shape experiments
should load this version without changing it. Any retraining with new odors,
hardware conditions, or protocol changes requires a new version and model card.

`temporal-mint-seeker-v1/deployment_policy.json` defines the required
hierarchical inference rule. Temporal active-mint evidence is retained only
when the frozen odor-presence model scores at least 0.5; otherwise active-mint
evidence is set to zero before spatial smoothing. Recovery remains ungated.
The same policy records the 2.0-second working spatial pose correction and the
fixed reported source center used by Notebook 06. These deployment settings
change no fitted model weights and preserve the original model checksum.
