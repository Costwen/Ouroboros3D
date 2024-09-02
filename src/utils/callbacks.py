from typing import Any, Dict, Optional

import os
import shutil
import lightning.pytorch as pl
from omegaconf import OmegaConf, DictConfig

from lightning import Trainer, Callback, LightningModule
from lightning_utilities.core.rank_zero import rank_zero_only

from .config import get_hparams, dump_config
from .misc import resolve_dir


import torch

class SavePathCallback(Callback):
    def __init__(self, dirpath: Optional[str] = None, sub_dir: Optional[str] = None):
        super().__init__()

        self.dirpath = dirpath
        self.sub_dir = sub_dir

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        dirpath = self.__resolve_dir(trainer)
        self.savedir = dirpath

    def __resolve_dir(self, trainer: Trainer) -> str:
        """Determines save directory at runtime. Reference attributes from the trainer's logger to
        determine where to save. The path for saving weights is set in this priority:

        1.  The ``Callback``'s ``dirpath`` if passed in
        2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
        3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

        The path gets extended with subdirectory ``sub_dir`` if passed in.

        """
        return resolve_dir(trainer, self.dirpath, self.sub_dir)


class VersionedCallback(Callback):
    def __init__(self, dirpath, version=None, use_version=True):
        self.dirpath = dirpath
        self._version = version
        self.use_version = use_version

    @property
    def version(self) -> int:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        existing_versions = []
        if os.path.isdir(self.dirpath):
            for f in os.listdir(self.dirpath):
                bn = os.path.basename(f)
                if bn.startswith("version_"):
                    dir_ver = os.path.splitext(bn)[0].split("_")[1].replace("/", "")
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0
        return max(existing_versions) + 1

    @property
    def savedir(self):
        if not self.use_version:
            return self.dirpath
        return os.path.join(
            self.dirpath,
            self.version
            if isinstance(self.version, str)
            else f"version_{self.version}",
        )


class ConfigSnapshotCallback(SavePathCallback):
    def __init__(
        self,
        config: DictConfig,
        config_path: Optional[str] = None,
        dirpath: Optional[str] = None,
        sub_dir: Optional[str] = "configs",
    ):
        super().__init__(dirpath, sub_dir)
        self.config = config
        self.config_path = config_path

    @rank_zero_only
    def save_config_snapshot(self, pl_module: LightningModule = None):
        os.makedirs(self.savedir, exist_ok=True)
        hparams = get_hparams(self.config)
        dump_config(os.path.join(self.savedir, "parsed.yaml"), hparams)

        if self.config_path:
            shutil.copy(self.config_path, os.path.join(self.savedir, "raw.yaml"))

    def on_fit_start(self, trainer, pl_module: LightningModule):
        self.save_config_snapshot(pl_module)


class PointCloudCallback(Callback):
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        filename = "point_cloud.ply"
        with torch.no_grad():
            output_dir = os.path.join(pl_module.save_dir, "point_cloud",
                                      "iteration_{}".format(trainer.global_step))
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            pl_module.save_ply(output_path + ".tmp")
            # os.rename(output_path + ".tmp", output_path)

        print("Gaussians saved to {}".format(output_path))