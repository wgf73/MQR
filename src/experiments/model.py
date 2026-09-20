from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from models.RAREMed import RAREMed

from .config import ExperimentConfig
from .plugin import ExperimentPlugin, PluginContext
from .registry import resolve_plugin



class PluginModel(nn.Module):
    """Composition layer that leaves the RAREMed implementation untouched."""

    def __init__(self, baseline: RAREMed, plugins: List[ExperimentPlugin]):
        super().__init__()
        self.baseline = baseline

    def _install_module_hooks(self) -> None:
        for plugin in self.plugins:

    def forward(self, model_input: Any, mode: str = "fine-tune"):
        transformed_input = model_input
        for plugin in self.plugins:
            transformed_input = plugin.transform_input(transformed_input, mode)

        for plugin in self.plugins:
            logits = plugin.transform_logits(logits, transformed_input, mode)

        if mode == "fine-tune":
            plugin_losses = [
                plugin.additional_loss(logits, transformed_input, mode)
                for plugin in self.plugins
            ]
            if any(not isinstance(loss, torch.Tensor) or loss.ndim != 0 for loss in plugin_losses):
                raise TypeError("Each plugin additional_loss must return a scalar torch.Tensor")
            self._last_plugin_loss = sum(plugin_losses, logits.new_zeros(()))
        else:

        if mode == "fine-tune":
            return logits, baseline_aux_loss
        return logits

    def plugin_loss(self) -> torch.Tensor:
        return self._last_plugin_loss


def build_experiment_model(
    args: Any,
    voc_size: Tuple[int, int, int],
    ddi_adj: Any,
) -> Tuple[nn.Module, ExperimentConfig]:
    """Build the exact baseline unless at least one plugin is enabled."""

    config = args.experiment
    baseline = RAREMed(args, voc_size, ddi_adj)
    context = PluginContext(args=args, voc_size=voc_size, ddi_adj=ddi_adj)
    plugins = [
        resolve_plugin(spec.name)(spec.params, context)
        for spec in config.active_plugins
    ]
    return PluginModel(baseline, plugins), config




def load_experiment_state_dict(model: nn.Module, state_dict: dict) -> None:
    """Load native baseline checkpoints into plugin runs and vice versa safely."""

    if any(key.startswith("baseline.") for key in state_dict):
        plugin_keys = [key for key in state_dict if key.startswith("plugins.")]
        if plugin_keys:
            raise RuntimeError(
                "This checkpoint contains experiment plugin parameters. "
                "Enable the same experiment config before loading it."
            )
    model.load_state_dict(state_dict)
