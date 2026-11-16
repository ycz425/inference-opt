from pydantic import BaseModel, ConfigDict
from typing import Literal
from pathlib import Path
import yaml


type BackendType = Literal['transformers']
type Precision = Literal['fp32', 'fp16', 'bf16', 'int8', 'int4']
type Device = Literal['cuda', 'mps']
type Dataset = Literal['synthetic', 'sharegpt']

class InferenceConfig(BaseModel):
    model_name: str
    backend: BackendType
    precision: Precision = 'fp32'
    device: Device
    compile: bool = False
    batch_size: int = 1

    model_config = ConfigDict(extra='forbid')


class WorkloadConfig(BaseModel):
    dataset: Dataset
    num_requests: int
    num_tokens: int | None = None
    seed: int = 42

    model_config = ConfigDict(extra='forbid')


class QualityConfig(BaseModel):
    reference: InferenceConfig | None
    workload: WorkloadConfig

    model_config = ConfigDict(extra='forbid')


class Config(BaseModel):
    inference_config: InferenceConfig
    workload_config: WorkloadConfig
    quality_config: QualityConfig | None

    output_tokens: int = 1024
    warmup_batches: int = 0
    
    
    model_config = ConfigDict(extra='forbid')


    def update_inference(self, **kwargs):
        for key in kwargs:
            setattr(self.inference_config, key, kwargs[key])


def load_config(path: str | Path) -> Config:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    return Config.model_validate(raw)   


def save_config(config: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        yaml.safe_dump(config.model_dump(mode='json'), f, sort_keys=False)