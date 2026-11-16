import torch
import threading


class TorchMPSMemoryMonitor:
    def __init__(self, interval=0.01):
        self.peak_memory = 0
        self.stop_event = threading.Event()
        self.interval = interval
        self.thread = None

    def _sample(self) -> int:
        while not self.stop_event.is_set():
            self.peak_memory = max(self.peak_memory, torch.mps.current_allocated_memory())
            self.stop_event.wait(self.interval)

    def reset(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.stop_event.clear()

        self.peak_memory = 0

        self.thread = threading.Thread(target=self._sample)
        self.thread.start()


    def get_peak_memory(self) -> int:
        self.stop_event.set()
        self.thread.join()

        return self.peak_memory

    def get_current_memory(self) -> int:
        return torch.mps.current_allocated_memory()