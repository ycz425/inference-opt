from typing import Protocol

class MemoryMonitor(Protocol):
    def reset(self) -> None:
        ...

    def get_current_memory(self) -> int:
        ...

    def get_peak_memory(self) -> int:
        ...