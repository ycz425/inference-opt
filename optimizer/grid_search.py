from .base import Optimizer, Grid
from .constraint import Constraint, constraint_satisfied
from itertools import product
from config import Config, load_config
from experiment import ExperimentRunner
from pathlib import Path


class GridSearch(Optimizer):
    def __init__(self, config: Config, output_dir: str | Path):
        super().__init__(config, output_dir)

    def search(self, grid: Grid, metric: str, constraint:  Constraint | None = None):
        grid = grid.model_dump()
        runner = ExperimentRunner(self.output_dir)
        best_metric = 0
        best_config = None
        for combination in product(*grid.values()):
            config = self.config.model_copy()
            config.update(**dict(zip(grid.keys(), combination)))

            try:
                result = runner.run(config)
            except:
                continue

            if not constraint_satisfied(result, constraint):
                continue

            if getattr(result, metric) > best_metric:
                best_metric = getattr(result, metric)
                best_config = config

        return best_config

            

grid = Grid(
    precision=['fp16', 'fp32', 'bf16', 'int4', 'int8'],
    batch_size=[1, 2, 4, 8, 16]
)

config = load_config('/Users/ycz425/Desktop/projects/inference-opt/config.yaml')
opt = GridSearch(config, 'test_optimization')
print(opt.search(grid))