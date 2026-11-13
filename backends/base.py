from typing import Protocol, Any
from workloads import Workload
from config import Precision, Device

class Backend(Protocol):
    def __init__(self, model_name: str, preicion: Precision, max_new_tokens: int, device: Device):
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

