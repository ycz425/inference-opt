from typing import Protocol, Any
from workloads import Workload

class Backend(Protocol):
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
    