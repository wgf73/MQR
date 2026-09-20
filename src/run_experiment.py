"""Launch one experiment on MIMIC-III and MIMIC-IV from a single YAML."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


ALLOWED_DATASET_OVERRIDES = {
    "records_file", "voc_file", "ddi_file", "data_root",
}




def enough_memory_for_parallel(
    free_memory_mib: int,
    job_count: int,
    memory_per_job_gib: float,
    reserve_gib: float,
) -> bool:
    required_mib = (job_count * memory_per_job_gib + reserve_gib) * 1024
    return free_memory_mib >= required_mib


def load_launcher(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as config_file:
    datasets = launcher.get("datasets", {"mimic-iii": {}, "mimic-iv": {}})
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("launcher.datasets must be a non-empty mapping or list")
    for dataset, overrides in datasets.items():
            raise ValueError(f"Unknown dataset overrides for {dataset}: {sorted(unknown)}")
    return launcher, datasets


def build_commands(config_path: Path, datasets: Dict[str, Dict[str, Any]], cuda: int) -> List[List[str]]:
    main_script = Path(__file__).with_name("main_RAREMed.py")
    commands = []
    for dataset, overrides in datasets.items():
        command = [
            dataset,
            "--cuda",
            str(cuda),
        ]
        for key, value in overrides.items():
            command.extend([f"--{key}", str(value)])
        commands.append(command)
    return commands


def main() -> int:
    launcher, datasets = load_launcher(config_path)
    execution = launcher.get("execution", "auto")
    if execution not in {"auto", "parallel", "serial"}:
        raise ValueError("launcher.execution must be auto, parallel, or serial")
    cuda = int(launcher.get("cuda", 0))
    commands = build_commands(config_path, datasets, cuda)
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
        processes = [subprocess.Popen(command, cwd=Path(__file__).parent) for command in commands]
        return_codes = [process.wait() for process in processes]
        return max(return_codes)

    for command in commands:
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
