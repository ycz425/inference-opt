from .base import Optimizer, Grid
from .constraint import Constraint, constraint_satisfied
from itertools import product
from config import Config, load_config
from experiment import ExperimentRunner
from pathlib import Path
from datetime import datetime


class GridSearch(Optimizer):
    def __init__(self, config: Config, output_dir: str | Path, verbose: bool = False):
        super().__init__(config, output_dir, verbose)

    def search(self, grid: Grid, metric: str, constraint:  Constraint | None = None):
        grid = grid.model_dump()
        runner = ExperimentRunner(self.output_dir)
        best_metric = 0
        best_config = None
        for combination in product(*grid.values()):
            if self.verbose:
                print(f'[{datetime.now()}] Starting {combination}')
            
            config = self.config.model_copy(deep=True)
            config.update_inference(**dict(zip(grid.keys(), combination)))

            result = runner.run(config)

            # try:
                
            # except:
            #     continue

            if not constraint_satisfied(result, constraint):
                continue

            if getattr(result, metric) > best_metric:
                best_metric = getattr(result, metric)
                best_config = config

        return best_config

            

grid = Grid(
    precision=['fp16', 'int8'],
    batch_size=[4, 8]
)

constraint = Constraint(
    max_ce_degradation=1.0,          
    max_memory=8 * 1024**3,         
    max_mean_e2e_latency=10.0
)

config = load_config('/Users/ycz425/Desktop/projects/inference-opt/config.yaml')
opt = GridSearch(config, 'test_optimization', verbose=True)
print(opt.search(grid, metric='output_tokens_per_sec', constraint=constraint))