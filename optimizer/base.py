from abc import ABC, abstractmethod
from pydantic import BaseModel
from config import Precision, Config
from pathlib import Path
from .constraint import Constraint


class Grid(BaseModel):
    precision: list[Precision]
    batch_size: list[int]


class Optimizer(ABC):
    def __init__(self, config: Config, output_dir: str | Path, verbose: bool = False):
        self.config = config  # Grid fields get overwritten
        self.output_dir = output_dir
        self.verbose = verbose

    @abstractmethod
    def search(self, grid: Grid, metric: str, constraint: Constraint | None = None) -> Config:
        ...
    