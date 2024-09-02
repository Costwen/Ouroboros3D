from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from omegaconf import OmegaConf, DictConfig


@dataclass
class ExperimentConfig:
    name: str = "default"
    tags: List[str] = field(default_factory=list)
    description: str = ""
    version: Optional[str] = None
    output_dir: str = "outputs/"

    seed: int = 42
    resume: Optional[str] = None

    data: Dict[str, Any] = field(default_factory=dict)
    system: Dict[str, Any] = field(default_factory=dict)
    trainer: Dict[str, Any] = field(default_factory=dict)
    callbacks: Optional[Dict[str, Any]] = None
    logger: Optional[Dict[str, Any]] = None

    train: bool = True
    test: bool = True

    extras: Optional[Dict[str, Any]] = None
    save_config: bool = True


def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg = parse_structured(ExperimentConfig, cfg)
    return scfg


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg


def get_hparams(cfg: DictConfig, model=None):
    hparams = OmegaConf.to_container(cfg)

    if model:
        # save number of model parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )

        hparams.update(
            {
                "model/params/total": total,
                "model/params/trainable": trainable,
                "model/params/non_trainable": non_trainable,
                "model/params/total_mb": total * 4 / 1024**2,
                "model/params/trainable_mb": trainable * 4 / 1024**2,
                "model/params/non_trainable_mb": non_trainable * 4 / 1024**2,
            }
        )

    return hparams
