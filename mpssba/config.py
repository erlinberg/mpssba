import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Union


@dataclass
class ModelConfig:
    n_qubits: int = 12
    bond_dim: int = 64


@dataclass
class MaskingConfig:
    type: str = "corr_2body"
    value: Union[int, List[int]] = 12


@dataclass
class TrainingConfig:
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    epochs: int = 1000
    tolerance: float = 1e-17
    record_every: int = 1
    hw_limits: List[int] = field(default_factory=list)


@dataclass
class DistConfig:
    method: str = "boltzmann"
    seed: int = 42
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InitConfig:
    method: str = "rnd_full"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    runs: int = 10
    postfix: str = "_run{run_idx}"
    run_continue: bool = False
    previous_run_path: str = ""


@dataclass
class OutputConfig:
    run_dir: str = ""
    raw_dir: str = ""
    processed_dir: str = ""
    plot_dir: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dist: DistConfig = field(default_factory=DistConfig)
    init: InitConfig = field(default_factory=InitConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict):
        def convert(dc_class, d):
            if not isinstance(d, dict):
                return d
            kwargs = {}
            fields = dc_class.__dataclass_fields__
            for k, v in d.items():
                if k in fields:
                    ftype = fields[k].type
                    if hasattr(ftype, "__dataclass_fields__"):
                        kwargs[k] = convert(ftype, v)
                    else:
                        kwargs[k] = v
            return dc_class(**kwargs)

        return convert(cls, data)

    def to_dict(self):
        return asdict(self)


RUN_ID_FIELDS = [
    ("model.n_qubits", "nq"),
    ("model.bond_dim", "bd"),
    ("batch.runs", "rn"),
    ("init.method", ""),
    ("dist.method", ""),
    ("dist.kwargs.target_entropy", "ent"),
    ("training.masking.type", ""),
    ("training.masking.value", ""),
]


def _generate_run_id(config: Config):
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_id = f"{timestamp}"

    for field_path, prefix in RUN_ID_FIELDS:
        keys = field_path.split(".")
        value = config
        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            elif isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = "x"
                break

        if isinstance(value, list):
            value = "_".join(map(str, value))

        run_id += f"_{prefix}{value}"

    return run_id


def load_config_from_json(json_path: str) -> Config:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config file {json_path} not found.")
    with open(json_path, "r") as f:
        config = Config.from_dict(json.load(f))
        return config
    

def setup_experiment_dir(config: Config) -> None:
    run_id = _generate_run_id(config)
    run_dir = os.path.join("results", "runs", run_id)

    config.output.run_dir = run_dir
    config.output.raw_dir = os.path.join(run_dir, "raw")
    config.output.processed_dir = os.path.join(run_dir, "processed")
    config.output.plot_dir = os.path.join(run_dir, "plots")

    os.makedirs(config.output.raw_dir, exist_ok=True)
    os.makedirs(config.output.processed_dir, exist_ok=True)
    os.makedirs(config.output.plot_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    return config
