import argparse
from workloads import workload_from_config
from backends import backend_from_config
from memory import memory_monitor_from_config
from benchmark import BenchmarkRunner, benchmark_result
from config import load_config, save_config
from pathlib import Path
from datetime import datetime



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, default='config.yaml')
    parser.add_argument('-e', '--experiment', type=Path, default='experiments')
    args = parser.parse_args()

    output_dir = args.experiment / f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
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


if __name__ == '__main__':
    main()







