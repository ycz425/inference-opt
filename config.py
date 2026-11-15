from pydantic import BaseModel
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
    precision: Precision
    batch_size: int
    device: Device


class WorkloadConfig(BaseModel):
    dataset: Dataset
    num_requests: int
    seed: int = 42


class QualityConfig(BaseModel):
    reference: InferenceConfig | None
    workload: WorkloadConfig
    num_tokens_per_request: int = 500


class Config(BaseModel):
    inference_config: InferenceConfig
    output_tokens: int
    input_tokens: int | None = None
    workload_config: WorkloadConfig
    warmup_batches: int = 3
    seed: int = 42

    quality_config: QualityConfig | None


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