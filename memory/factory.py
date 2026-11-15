from config import Config, BackendType, Device
from .base import MemoryMonitor
from .torch_mps import TorchMPSMemoryMonitor
from .torch_cuda import TorchCUDAMemoryMonitor

MEMORY_MONITORS: dict[tuple[BackendType, Device], MemoryMonitor] = {
    ('transformers', 'cuda'): TorchCUDAMemoryMonitor(),
    ('transformers', 'mps'): TorchMPSMemoryMonitor()
}

def memory_monitor_from_config(config: Config):
    return MEMORY_MONITORS[(config.inference_config.backend, config.inference_config.device)]