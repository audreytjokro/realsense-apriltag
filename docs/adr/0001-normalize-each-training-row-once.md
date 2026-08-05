# Normalize each training physical row once

Smell-sequence models use normalization statistics in which every training physical row contributes exactly once, while validation and held-out-session test rows contribute nothing. Repeated sliding-window occurrences previously gave block-interior rows length-dependent weight, making normalization itself vary during sequence-length ablations; accepting equal row weight requires retraining the canonical baseline rather than relabeling existing checkpoints.
