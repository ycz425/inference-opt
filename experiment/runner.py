from pathlib import Path
from workload import workload_from_config
from backends import backend_from_config
from memory import memory_monitor_from_config
from benchmark import BenchmarkRunner, benchmark_result, BenchmarkResult
from transformers import AutoTokenizer
from config import save_config, Config, Device
from pathlib import Path
from datetime import datetime
import torch
import gc


class ExperimentRunner:
    def __init__(self, experiment_dir: str | Path):
        self.experiment_dir = Path(experiment_dir)

    def run(self, config: Config) -> BenchmarkResult:
        output_dir = self.experiment_dir / f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        output_dir.mkdir(parents=True, exist_ok=True)
    
        save_config(config, output_dir / 'config.yaml')

        tokenizer = AutoTokenizer.from_pretrained(config.inference_config.model_name)
    
        workload = workload_from_config(config.workload_config, tokenizer)
        backend = backend_from_config(config.inference_config)
        memory_monitor = memory_monitor_from_config(config)

        if memory_monitor:
            model_memory = memory_monitor.get_current_memory()
        else:
            model_memory = None
    
        runner = BenchmarkRunner(backend=backend, memory_monitor=memory_monitor)
    
        batch_results, peak_memory, ce_degradation = runner.run(workload, config=config, tokenizer=tokenizer)

        result = benchmark_result(batch_results, peak_memory, model_memory, ce_degradation)
            
        output_path = output_dir / 'result.json'
        with output_path.open('w') as f:
            f.write(result.model_dump_json(indent=4))

        backend.free_device_memory()

        return result
