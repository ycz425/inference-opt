from .sharegpt import ShareGPTLoader
from config import Dataset, WorkloadConfig
from .base import WorkloadLoader

WORKLOAD_LOADERS: dict[Dataset, WorkloadLoader] = {
    'sharegpt': ShareGPTLoader()
}

def workload_from_config(config: WorkloadConfig):
    workload_loader = WORKLOAD_LOADERS[config.dataset]
    return workload_loader.load_workload(num_requests=config.num_requests, seed=config.seed)