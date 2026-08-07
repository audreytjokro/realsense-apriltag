"""Bidirectional GRU vs. the existing bidirectional Temporal CNN, one controlled variable:
temporal convolution vs. recurrent memory, on the identical 5 July-30 sessions, LOSO area task.

Reuses analysis.spatial_sequence's data pipeline (build_prepared_data, the Dataset classes,
loss, metrics, checkpoint utilities) unchanged; only the model class is new. See README.md.
"""

from .models import BiGRUAreaModel, count_parameters
from .training import RunConfig, evaluate_checkpoint, train_run

__all__ = [
    "BiGRUAreaModel",
    "RunConfig",
    "count_parameters",
    "evaluate_checkpoint",
    "train_run",
]
