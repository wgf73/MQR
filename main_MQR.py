#!/usr/bin/env python3
"""Simple command-line entry point for rebuilding MQR from RAREMed backbones."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from src.mqr_pipeline.ablations import ABLATIONS


DATASETS = ("mimic-iii", "mimic-iv")
ALL_STEPS = (
    "surface",
    "huatuo-preflight",
    "huatuo-extract",
    "huatuo-pool",
    "train-preflight",
    "train",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        add_help=False,
        description=(
            "Rebuild surfaces, Huatuo embeddings, and MQR from the selected "
            "RAREMed backbone. Exactly one action flag is required."
        ),
    )
    parser.add_argument(
        "-h", "-help", "--help", action="help",
        help="show this help message and exit",
    )
    parser.add_argument(
        "-dataset", "--dataset",
        choices=DATASETS,
        required=True,
        help="dataset and matching backbone/config directory",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "-surface", "--surface",
        dest="action", action="store_const", const="surface",
        help="rebuild RAREMed train/validation/test surfaces",
    )
    action.add_argument(
        "-huatuo_preflight", "--huatuo-preflight",
        dest="action", action="store_const", const="huatuo-preflight",
        help="validate the Huatuo snapshot and new surface inputs",
    )
    action.add_argument(
        "-huatuo_extract", "--huatuo-extract",
        dest="action", action="store_const", const="huatuo-extract",
        help="encode clinical-code and medication knowledge prompts",
    )
    action.add_argument(
        "-huatuo_pool", "--huatuo-pool",
        dest="action", action="store_const", const="huatuo-pool",
        help="pool Huatuo evidence into split visit/drug embeddings",
    )
    action.add_argument(
        dest="action", action="store_const", const="train-preflight",
        help="validate all rebuilt inputs without starting training",
    )
    parser.add_argument(
        "-dry_run", "--dry-run",
        help="print resolved commands without executing them",
    )
    parser.add_argument(
        "-ablation", "--ablation",
        choices=ABLATIONS,
        default="full",
    )
    args = parser.parse_args(argv)
    if args.ablation != "full" and args.action not in ("train", "train-preflight"):
        parser.error("-ablation can only be used with -train or -train_preflight")
    return args


def command_for(
    repo: Path, dataset: str, step: str, ablation: str = "full"
) -> list[str]:
    config_root = repo / "configs" / dataset
    python = str(Path(sys.executable).resolve())
    if step == "surface":
        ]
    if step.startswith("huatuo-"):
        return [
            python, "-m", "src.tkq_pipeline.huatuo", action,
        ]
    if step in ("train-preflight", "train"):
        command = [
        if ablation != "full":
            )
            command.extend([
                "--ablation", ablation,
            ])
        if step == "train-preflight":
            command.append("--preflight")
        return command
    raise ValueError(f"unsupported step: {step}")


def commands_for(
    repo: Path, dataset: str, action: str, ablation: str = "full"
) -> list[list[str]]:
    return [command_for(repo, dataset, step, ablation) for step in steps]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(__file__).resolve().parent
    commands = commands_for(repo, args.dataset, args.action, args.ablation)
    environment = dict(os.environ)
    required_python_path = (str(repo), str(repo / "src"))
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        (*required_python_path, existing) if existing else required_python_path
    )

    print(f"dataset: {args.dataset}", flush=True)
    print(f"action: {args.action}", flush=True)
    print(f"ablation: {args.ablation}", flush=True)
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=repo, env=environment, check=True)
    if args.dry_run:
        print("dry_run: no command was executed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
