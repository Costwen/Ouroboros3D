from typing import Any, Dict, Tuple, List, Optional

import os
from lightning import LightningModule

from src.utils.misc import load_module_weights, resolve_dir


class BaseSystem(LightningModule):
    def __init__(
        self,
        save_dir: Optional[str] = None,
        weights: Optional[str] = None,
        weights_ignore_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.save_dir = save_dir

        if weights is not None:
            self.load_weights(weights, ignore_modules=weights_ignore_modules)

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict, epoch, global_step = load_module_weights(
            weights, ignore_modules=ignore_modules, map_location="cpu"
        )
        self.load_state_dict(state_dict, strict=False)

    def setup(self, stage: str) -> None:
        # set and make save dir for saving validation results, etc.
        save_dir = resolve_dir(self.trainer, self.save_dir, "save")
        save_dir = self.trainer.strategy.broadcast(save_dir)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
