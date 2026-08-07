"""A single new architecture: a bidirectional GRU, held to the same input/output contract as
analysis.spatial_sequence.models.SpatialSequenceModel so training/evaluation code is reusable.
"""

from __future__ import annotations

import torch
from torch import nn

D_MODEL = 96
GRU_HIDDEN_SIZE = 64
GRU_LAYERS = 1

# v2: capacity roughly matched to the existing TCN (254,787 params), plus dropout and
# a LayerNorm on the GRU's output -- see analysis/gru_sequence/README.md for why these
# specific additions (and not a residual connection, at least not yet).
GRU_V2_HIDDEN_SIZE = 161
GRU_V2_DROPOUT = 0.1


class BiGRUAreaModel(nn.Module):
    """v1, unchanged: the original single-variable-isolation model (hidden=64, no
    dropout, no normalization). Left exactly as it was for the first experiment's
    results to stay reproducible -- v2 below is a separate class, not a parameter
    change to this one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(32, D_MODEL)
        self.gru = nn.GRU(
            input_size=D_MODEL,
            hidden_size=GRU_HIDDEN_SIZE,
            num_layers=GRU_LAYERS,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Linear(2 * GRU_HIDDEN_SIZE, 3)
        self.apply(self._initialize_module)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.input_projection(inputs)
        sequence, _ = self.gru(values)
        return {"area": self.head(sequence)}


class BiGRUAreaModelV2(nn.Module):
    """input_projection(32->96) [same as v1 and as the TCN/Transformer]
    -> single-layer bidirectional GRU (hidden=161/direction by default, ~matching the
       TCN's parameter count; 322-dim concat output)
    -> LayerNorm on the GRU output (the single-block analogue of the TCN's pre-norm)
    -> Dropout (matching the existing models' dropout=0.1, previously absent here
       since num_layers=1 makes nn.GRU's own internal dropout inert)
    -> linear head to 3 area classes, at every timestep (dense supervision, unchanged).
    No residual connection -- deferred; see README for the shape-mismatch reasoning.
    """

    def __init__(
        self,
        hidden_size: int = GRU_V2_HIDDEN_SIZE,
        dropout: float = GRU_V2_DROPOUT,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(32, D_MODEL)
        self.gru = nn.GRU(
            input_size=D_MODEL,
            hidden_size=hidden_size,
            num_layers=GRU_LAYERS,
            bidirectional=True,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(2 * hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_size, 3)
        self.apply(self._initialize_module)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.input_projection(inputs)
        sequence, _ = self.gru(values)
        sequence = self.output_norm(sequence)
        sequence = self.output_dropout(sequence)
        return {"area": self.head(sequence)}


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
