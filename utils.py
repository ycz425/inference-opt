from preprocessing import Inputs
import numpy as np
import torch

def count_input_tokens(inputs: Inputs) -> int:
    return np.array(inputs.attention_mask).sum(dtype=int)

def count_output_tokens(outputs: torch.LongTensor, pad_token_id: int):
    return (outputs != pad_token_id).sum(dtype=torch.int64).item()