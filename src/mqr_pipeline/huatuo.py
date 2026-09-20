"""Extract pinned Huatuo states and align pooled embeddings to T-KQ splits."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from tqdm.auto import tqdm



def _pool_visits(
    code: torch.Tensor,
    description: str,
) -> torch.Tensor:
    rows = range(len(offsets) - 1)
    values = [
        code[torch.from_numpy(indices[offsets[row] : offsets[row + 1]])].mean(0)
        for row in tqdm(
            rows,
        )
    ]
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError(f"non-finite pooled Huatuo visit: {surface}")
    return torch.stack(values).to(torch.float16)


def pool(config_path: Path) -> None:
    repo, config = _paths(config_path)
    if llm_root.exists():
        raise FileExistsError(llm_root)

    for dataset in configured_datasets(config):
        evidence_path = evidence_root / f"{dataset}.pt"
        evidence = torch.load(evidence_path, map_location="cpu", weights_only=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "extract", "pool"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    {"preflight": preflight, "extract": extract, "pool": pool}[args.action](
        args.config
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
