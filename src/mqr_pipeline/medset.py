"""Shared data, loss, and evaluation primitives for the canonical MQR path."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence


DATASETS = ("mimic-iii", "mimic-iv")


class EvidenceSlots(nn.Module):
    """Perceiver-style latent slots over all current-visit code tokens."""

    def __init__(self, dimension: int, heads: int, slots: int, dropout: float):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(slots, dimension) / math.sqrt(dimension))
        self.cross = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
      
        self.refine = nn.TransformerEncoder(layer, 1)
        self.norm = nn.LayerNorm(dimension)

    def forward(
        self,
    ) -> torch.Tensor:
        slots = self.seed.unsqueeze(0).expand(len(memory), -1, -1)
        return self.norm(self.refine(slots + update))


def load(config: dict, dataset: str) -> dict:
    surface = Path(config["data"]["surface_root"]) / dataset
    llm = torch.load(
        weights_only=False,
    )
    data = {
        key: np.load(surface / f"{key}.npy").astype("float32")
        for key in ("labels", "native_logits", "cls_states", "classifier_weight")
    }
    data["native"] = data.pop("native_logits")
    data["state"] = data.pop("cls_states")
    data["classifier"] = data.pop("classifier_weight")
    return data


def token_batch(data: dict, rows: torch.Tensor, device: torch.device):
    states, roles = [], []
    for row in rows.tolist():
        torch.arange(state_pad.shape[1], device=device).unsqueeze(0)
        >= lengths.unsqueeze(1)
    )
    return state_pad, role_pad, padding


def loss_fn(output: torch.Tensor, labels: torch.Tensor, config: dict) -> torch.Tensor:
    probability = output.sigmoid()
    intersection = (probability * labels).sum(1)
        + float(weights["bce"]) * bce
        + float(weights["cardinality"]) * cardinality
    )


def attach_evaluation_arrays(
    data: dict,
) -> None:
    """Load arrays used only by metric reporting, never training."""


def patient_means(values: np.ndarray, row_patient: np.ndarray) -> np.ndarray:
    return np.asarray(
    )


def metrics(logits: np.ndarray, data: dict) -> dict:
    truth = data["labels"] > 0.5
    prediction = logits >= 0.0
    intersection = np.logical_and(prediction, truth).sum(1)
    f1 = np.divide(
        2 * intersection,
        denominator,
        out=np.zeros(len(truth)),
        where=denominator > 0,
    )
    probability = 1.0 / (
        1.0 + np.exp(-np.clip(logits.astype(np.float64), -80, 80))
    )
    prauc = np.asarray([
        average_precision_score(truth[row].astype(np.int8), probability[row])
        for row in range(len(truth))
    ])
    ddi, pairs, unsafe = data["ddi"], 0, 0
    for row in prediction:
        medications = np.flatnonzero(row)
        pairs += len(medications) * (len(medications) - 1) // 2
        unsafe += sum(
            ddi[medications[left], medications[right]] > 0
            for left in range(len(medications))
            for right in range(left + 1, len(medications))
        )
    return {
        "patient_jaccard": float(patient_means(jaccard, data["row_patient"]).mean()),
        "patient_f1": float(patient_means(f1, data["row_patient"]).mean()),
        "patient_prauc": float(patient_means(prauc, data["row_patient"]).mean()),
        "ddi_rate": float(unsafe / pairs if pairs else 0.0),
        "mean_medications": float(prediction.sum(1).mean()),
    }
