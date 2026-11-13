from .sharegpt import ShareGPTLoader
from config import Dataset, Config
from .base import WorkloadLoader

WORKLOAD_LOADERS: dict[Dataset, WorkloadLoader] = {
    'sharegpt': ShareGPTLoader()
}

def workload_from_config(config: Config):
    workload_loader = WORKLOAD_LOADERS[config.dataset]
    return workload_loader.load_workload(size=config.workload_size, seed=config.seed)