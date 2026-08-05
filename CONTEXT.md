# Smell-Only Spatial Sequence Experiments

This context defines the evaluation language for predicting spatial and motion
properties from sequences of smell-sensor readings.

## Language

**Within-session validation**:
Evaluation on held-out physical rows from sessions that otherwise contribute training data. It may select checkpoints, so it is not an independent performance estimate.
_Avoid_: Test set, independent test

**Held-out-session test**:
Evaluation on complete sessions excluded from training, normalization, checkpoint selection, and hyperparameter selection. It measures generalization to a new recording session.
_Avoid_: Validation set, pooled validation

**Held-out-session validation**:
Evaluation on a complete session excluded from training and normalization but used for checkpoint selection and performance reporting. It is a cross-session validation estimate, not an independent test.
_Avoid_: Test set, independent test

**Leave-one-session-out evaluation**:
A cross-session evaluation in which one complete session is the held-out-session validation set and all remaining sessions supply all training rows, repeated once for every session.
_Avoid_: Independent final test, within-session validation

**Sequence-length ablation**:
A comparison that varies smell-reading context length while holding the remaining experimental protocol fixed.
_Avoid_: Sequence optimization, context tuning

**Training-row normalization**:
Input and target statistics computed from training physical rows, with every row contributing exactly once. Validation and test rows never contribute.
_Avoid_: Window-occurrence normalization, repeated-occurrence normalization
