from typing import Protocol, Any
from preprocessing import Inputs
from config import Precision, Device
import torch

class Backend(Protocol):
    def __init__(self, model_name: str, precision: Precision, device: Device):
        ...

    def generate(self, inputs: Inputs) -> torch.LongTensor:
        ...

    def detokenize(self, outputs: torch.LongTensor) -> list[str]:
        ...

    def synchronize(self) -> None:
        ...

    def compute_token_loss(self, inputs: Inputs) -> float:
        ...

