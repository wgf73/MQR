"""Replay locked RAREMed checkpoints into split surfaces without LLM inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dill
import numpy as np
import torch

from experiments.model import build_experiment_model, load_experiment_state_dict
from main_RAREMed import get_args
from src.mqr_pipeline.surface_encoding import encode_split, split_records
from src.mqr_pipeline.contracts import (
    configured_datasets,
    file_record,
    read_yaml,
    repo_path,
    sha256_file,
)

