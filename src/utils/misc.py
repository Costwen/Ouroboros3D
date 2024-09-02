import os
import re
import warnings
from importlib.util import find_spec
from typing import Callable, Optional, Tuple

import torch
from lightning import Trainer
from omegaconf import DictConfig

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def load_module_weights(
    path, module_name=None, ignore_modules=None, map_location=None
) -> Tuple[dict, int, int]:
    """Load module weights from a checkpoint.
    This method is useful when you want to load weights from a checkpoint. You can specify
    a module name, and only the weights of that module will be loaded. You can also specify
    a list of modules to ignore.

    :param path: Path to the checkpoint.
    :param module_name: Name of the module to load.
    :param ignore_modules: List of modules to ignore.
    :param map_location: Map location for the checkpoint. Defaults to the current device.
    :return: A tuple containing the state dict, the epoch and the global step.
    """
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        map_location = get_device()

    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["state_dict"]
    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any(
                [k.startswith(ignore_module + ".") for ignore_module in ignore_modules]
            )
            if ignore:
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            state_dict_to_load[m.group(1)] = v

    return state_dict_to_load, ckpt["epoch"], ckpt["global_step"]


def resolve_dir(
    trainer: Trainer,
    dirpath: Optional[str] = None,
    sub_dir: Optional[str] = None,
) -> str:
    """Determines save directory at runtime. Reference attributes from the trainer's logger to
    determine where to save. The path for saving weights is set in this priority:

    1.  The ``dirpath`` if passed in
    2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
    3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

    The path gets extended with subdirectory ``name`` if passed in.

    """
    if dirpath is not None:
        # short circuit if dirpath was passed to ModelCheckpoint
        return dirpath

    if len(trainer.loggers) > 0:
        if trainer.loggers[0].save_dir is not None:
            save_dir = trainer.loggers[0].save_dir
        else:
            save_dir = trainer.default_root_dir
        _name = trainer.loggers[0]._name
        version = trainer.loggers[0].version
        version = version if isinstance(version, str) else f"version_{version}"
        dirpath = os.path.join(save_dir, _name or version)
    else:
        # if no loggers, use default_root_dir
        dirpath = trainer.default_root_dir

    if sub_dir is not None:
        dirpath = os.path.join(dirpath, sub_dir)

    return dirpath


def apply_extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - ...

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(**kwargs):
        ...
        return
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg, config_path=None):
        # execute the task
        try:
            task_func(cfg, config_path=config_path)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

    return wrap
