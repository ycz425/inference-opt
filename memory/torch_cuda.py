import torch

class TorchCUDAMemoryMonitor:
    def reset(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def get_peak_memory(self) -> int:
        return torch.cuda.max_memory_allocated()