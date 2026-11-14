from typing import Protocol, Any
from workload import Workload
from config import Precision, Device

class Backend(Protocol):
    def __init__(self, model_name: str, precision: Precision, output_tokens: int, device: Device, seed: int):
        ...

    def preprocess(self, workload: Workload) -> Any:
        ...

    def generate(self, inputs: Any) -> Any:
        ...

    def detokenize(self, outputs: Any) -> list[str]:
        ...

    def count_input_tokens(self, inputs: Any) -> int:
        ...

    def count_output_tokens(self, outputs: Any) -> int:
        ...

    def synchronize(self) -> None:
        ...

