from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


ModuleHook = Callable[[nn.Module, Tuple[Any, ...], Any], Any]


class ExperimentPlugin(nn.Module):
    """Extension points around an unchanged baseline model.

    A plugin may override only the methods it needs. Plugins are executed in the
    order in which they appear in the experiment config.
    """

    def __init__(self, params: Mapping[str, Any], context: PluginContext):
        super().__init__()
        self.params = dict(params)
        self.context = context


    def additional_loss(
    ) -> torch.Tensor:

        return logits.new_zeros(())
