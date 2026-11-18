from abc import ABC, abstractmethod
from pydantic import BaseModel
from config import Precision, Config
from pathlib import Path
from .constraint import Constraint
import yaml


class Grid(BaseModel):
    precision: list[Precision]
    batch_size: list[int]
    compile: list[bool]


class Optimizer(ABC):
    def __init__(self, config: Config, output_dir: str | Path, verbose: bool = False):
        self.config = config  # Grid fields get overwritten
        self.output_dir = output_dir
        self.verbose = verbose

    @abstractmethod
    def search(self, grid: Grid, metric: str, constraint: Constraint | None = None) -> Config:
        ...


def load_grid(path: Path | str):
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return Grid.model_validate(raw)   