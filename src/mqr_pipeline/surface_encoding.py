"""RAREMed visit replay used to build deterministic T-KQ surfaces."""
from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm


def split_records(records):
    return {
        "validation": records[train_count : train_count + validation_count],
        "test": records[train_count + validation_count :],
    }


@torch.inference_mode()
def encode_split(
    model,
    *,
    description: str = "RAREMed surfaces",
):
    cls_states, native_logits, labels, visit_weights = [], [], [], []
    token_states, roles, code_indices, offsets = [], [], [], [0]
    row_patient = []
    for patient_index, patient in enumerate(progress):
        for visit in patient:
            diagnoses = list(map(int, visit[0]))
            target = np.zeros(medications, dtype=np.float32)
            target[np.asarray(visit[2], dtype=np.int64)] = 1.0
            offsets.append(offsets[-1] + len(diagnoses) + len(procedures))
            row_patient.append(patient_index)
    return {
        "cls_states": np.asarray(cls_states, dtype=np.float32),
        "classifier_weight": model.cls_final.weight.detach().cpu().float().numpy(),
        "classifier_bias": model.cls_final.bias.detach().cpu().float().numpy(),
    }
