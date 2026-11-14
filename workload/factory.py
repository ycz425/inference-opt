from .sharegpt import ShareGPTLoader
from .synthetic import SyntheticLoader
from config import Dataset, Config
from .base import WorkloadLoader

WORKLOAD_LOADERS: dict[Dataset, WorkloadLoader] = {
    'sharegpt': ShareGPTLoader(),
    'synthetic': SyntheticLoader()
}

def workload_from_config(config: Config):
    workload_loader = WORKLOAD_LOADERS[config.dataset]
    return workload_loader.load_workload(size=config.workload_size, input_tokens=config.input_tokens, seed=config.seed)