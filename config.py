from pydantic import BaseModel
from typing import Literal
from pathlib import Path
import yaml


type BackendType = Literal['transformers']
type Precision = Literal['fp32', 'fp16', 'bf16', 'int8', 'int4']
type Device = Literal['cuda', 'mps']
type Dataset = Literal['synthetic', 'sharegpt']

class Config(BaseModel):
    model_name: str
    backend: BackendType
    precision: Precision = 'fp32'
    input_tokens: int | None = None  # if dataset = 'synthetic', must be set to an int value
    output_tokens: int = 256
    device: Device
    dataset: Dataset
    workload_size: int
    batch_size: int = 1
    warmup_batches: int = 3
    seed: int = 42

    def update(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])


def load_config(path: str | Path) -> Config:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    return Config.model_validate(raw)   


def save_config(config: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        yaml.safe_dump(config.model_dump(mode='json'), f, sort_keys=False)