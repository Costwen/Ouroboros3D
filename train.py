import argparse
import os
from typing import List

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    ExperimentConfig,
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    load_config,
    log_hyperparameters,
)
from src.utils.callbacks import ConfigSnapshotCallback
from src.utils.misc import apply_extras, get_rank, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: ExperimentConfig, config_path=None):
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by instantiator.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    L.seed_everything(cfg.seed + get_rank(), workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating modelmodule <{cfg.system._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.system)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.callbacks)

    if cfg.save_config:
        callbacks += [ConfigSnapshotCallback(cfg, config_path)]

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.logger)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # FIXME: hard code
    model.strict_loading = False

    if cfg.train:
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.resume)

    train_metrics = trainer.callback_metrics
    log.info(f"Train metrics: {train_metrics}")

    if cfg.test:
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            if cfg.resume is not None:
                ckpt_path = cfg.resume
            else:
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    log.info(f"Test metrics: {test_metrics}")


def main(args, extras):
    """Main entry point for training.

    :param args: Arguments passed from command line.
    :param extras: Arguments passed from command line, but not recognized by argparse.
    """
    # load config from file and cli args
    cfg: ExperimentConfig = load_config(args.config, cli_args=extras)

    # apply extra utilities
    # (e.g. ask whether to ignore warning, etc.)
    apply_extras(cfg)

    # train the model
    train(cfg, args.config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    args, extras = parser.parse_known_args()

    main(args, extras)
