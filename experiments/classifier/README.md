# Classifier experiments

This stage contains stationary PCnose+ trials used to learn odor-response and
odor-identity models. Each pilot keeps its own raw copies, manifest, protocol
notes, and analysis scope so later odor panels can be added without changing
earlier results.

Current pilots:

- `mint-identity-pilot-01/`: blank, mint, bergamot, and lemongrass stationary
  trials used by `analysis/notebooks/05_mint_identity_classifier.ipynb`.
- `mint-lavender-ambient-pilot-01/`: balanced 45-trial closed panel with
  15 ambient, 15 mint, and 15 lavender trials used by
  `analysis/notebooks/07_mint_lavender_ambient_classifier.ipynb`. Its primary
  hierarchical model achieved 15/15 held-out recall for every class and is
  frozen as `mint-lavender-ambient-v1`. Lavender was collected on a later date
  than the existing classes, so the result is not yet session-invariant and
  does not establish mixture recognition.

Stationary files train the chemical fingerprint model. Spatial mapping sessions
remain under `experiments/spatial-mapping/` and are used as transfer tests.
