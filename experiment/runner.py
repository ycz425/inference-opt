from pathlib import Path
from workload import workload_from_config
from backends import backend_from_config
from memory import memory_monitor_from_config
from benchmark import BenchmarkRunner, benchmark_result, BenchmarkResult
from config import load_config, save_config, Config
from pathlib import Path
from datetime import datetime


class ExperimentRunner:
    def __init__(self, experiment_dir: str | Path):
        self.experiment_dir = Path(experiment_dir)

    def run(self, config: Config) -> BenchmarkResult:
        output_dir = self.experiment_dir / f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        output_dir.mkdir(parents=True, exist_ok=True)
    
        save_config(config, output_dir / 'config.yaml')
    
        workload = workload_from_config(config)
        backend = backend_from_config(config)
        memory_monitor = memory_monitor_from_config(config)
    
        runner = BenchmarkRunner(backend, memory_monitor)
    
        batch_results, peak_memory = runner.run(workload, config=config)
    
        result = benchmark_result(batch_results, peak_memory)
    
        output_path = output_dir / 'result.json'
        with output_path.open('w') as f:
            f.write(result.model_dump_json(indent=4))

        return result