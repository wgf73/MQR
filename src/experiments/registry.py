import importlib
from typing import Dict, Type

from .plugin import ExperimentPlugin


_REGISTRY: Dict[str, Type[ExperimentPlugin]] = {}


def register_plugin(name: str):
    """Register a short plugin name without coupling it to the baseline."""


def resolve_plugin(name: str) -> Type[ExperimentPlugin]:
    """Resolve a built-in short name or ``python.module:ClassName`` reference."""

    # Importing built-ins here keeps registration lazy and avoids import cycles.