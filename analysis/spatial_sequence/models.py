from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .core import POSITION_CLASS_COUNT


D_MODEL = 96
BLOCK_COUNT = 3
FFN_DIM = 192
HEAD_COUNT = 4
DROPOUT = 0.1
CNN_KERNEL_SIZE = 5
CNN_DILATIONS = (1, 2, 3)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 24) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32)[:, None]
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("encoding", encoding, persistent=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.encoding.shape[0]:
            raise ValueError("Sequence exceeds the fixed positional encoding length")
        return values + self.encoding[: values.shape[1]].unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(D_MODEL, FFN_DIM)
        self.activation = nn.GELU()
        self.hidden_dropout = nn.Dropout(DROPOUT)
        self.second = nn.Linear(FFN_DIM, D_MODEL)
        self.output_dropout = nn.Dropout(DROPOUT)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.first(values)
        values = self.activation(values)
        values = self.hidden_dropout(values)
        values = self.second(values)
        return self.output_dropout(values)


class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(D_MODEL)
        self.attention = nn.MultiheadAttention(
            D_MODEL,
            HEAD_COUNT,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(DROPOUT)
        self.ffn_norm = nn.LayerNorm(D_MODEL)
        self.ffn = FeedForward()

    def forward(
        self,
        values: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        values = values + self.attention_dropout(attended)
        return values + self.ffn(self.ffn_norm(values))


class TemporalCNNBlock(nn.Module):
    def __init__(self, dilation: int, causal: bool) -> None:
        super().__init__()
        self.convolution_norm = nn.LayerNorm(D_MODEL)
        self.convolution = nn.Conv1d(
            D_MODEL,
            D_MODEL,
            kernel_size=CNN_KERNEL_SIZE,
            dilation=dilation,
            padding=0,
        )
        total_padding = dilation * (CNN_KERNEL_SIZE - 1)
        self.padding = (total_padding, 0) if causal else (
            total_padding // 2,
            total_padding // 2,
        )
        self.convolution_dropout = nn.Dropout(DROPOUT)
        self.ffn_norm = nn.LayerNorm(D_MODEL)
        self.ffn = FeedForward()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.convolution_norm(values).transpose(1, 2)
        convolved = self.convolution(F.pad(normalized, self.padding)).transpose(1, 2)
        values = values + self.convolution_dropout(convolved)
        return values + self.ffn(self.ffn_norm(values))


class SpatialSequenceModel(nn.Module):
    def __init__(
        self,
        architecture: str,
        temporal_mode: str,
        target: str,
        maximum_sequence_length: int = 24,
    ) -> None:
        super().__init__()
        if architecture not in {"transformer", "temporal-cnn"}:
            raise ValueError(f"Unknown architecture: {architecture}")
        if temporal_mode not in {"causal", "bidirectional"}:
            raise ValueError(f"Unknown temporal mode: {temporal_mode}")
        if target not in {"position", "distance", "area", "height", "velocity"}:
            raise ValueError(f"Unknown target: {target}")
        self.architecture = architecture
        self.temporal_mode = temporal_mode
        self.target = target
        self.input_projection = nn.Linear(32, D_MODEL)
        if architecture == "transformer":
            self.position_encoding: SinusoidalPositionEncoding | None = (
                SinusoidalPositionEncoding(D_MODEL, maximum_sequence_length)
            )
            self.blocks = nn.ModuleList(TransformerBlock() for _ in range(BLOCK_COUNT))
        else:
            self.position_encoding = None
            self.blocks = nn.ModuleList(
                TemporalCNNBlock(dilation, temporal_mode == "causal")
                for dilation in CNN_DILATIONS
            )
        self.final_norm = nn.LayerNorm(D_MODEL)
        if target == "position":
            self.heads = nn.ModuleDict(
                {"position": nn.Linear(D_MODEL, POSITION_CLASS_COUNT)}
            )
        elif target == "area":
            self.heads = nn.ModuleDict({"area": nn.Linear(D_MODEL, 3)})
        elif target in {"distance", "velocity"}:
            self.heads = nn.ModuleDict({target: nn.Linear(D_MODEL, 2)})
        else:
            self.heads = nn.ModuleDict({"height": nn.Linear(D_MODEL, 1)})
        self.apply(self._initialize_module)
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                nn.init.trunc_normal_(module.in_proj_weight, std=0.02)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor | None:
        if self.temporal_mode != "causal":
            return None
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.input_projection(inputs)
        if self.architecture == "transformer":
            assert self.position_encoding is not None
            values = self.position_encoding(values)
            attention_mask = self._causal_mask(values.shape[1], values.device)
            for block in self.blocks:
                assert isinstance(block, TransformerBlock)
                values = block(values, attention_mask)
        else:
            for block in self.blocks:
                assert isinstance(block, TemporalCNNBlock)
                values = block(values)
        values = self.final_norm(values)
        outputs = {name: head(values) for name, head in self.heads.items()}
        if self.target in {"distance", "height"}:
            outputs[self.target] = F.softplus(outputs[self.target])
        return outputs


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    layer_norm_parameters: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            layer_norm_parameters.update(id(parameter) for parameter in module.parameters())
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("bias") or id(parameter) in layer_norm_parameters:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
