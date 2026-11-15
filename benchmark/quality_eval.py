from backends import Backend
from workload import Workload
from preprocessing import preprocess
from transformers import PreTrainedTokenizerBase
from utils import count_input_tokens
import numpy as np

class QualityEvaluator:
    def __init__(self, candidate: Backend, reference: Backend, tokenizer: PreTrainedTokenizerBase):
        self.candidate = candidate
        self.reference = reference
        self.tokenizer = tokenizer

    def run(self, workload: Workload, num_tokens_per_request: int, batch_size: int) -> float:
        candidate_losses = []
        reference_losses = []
        num_tokens = []

        for i in range(0, len(workload.requests), batch_size):
            batch = Workload(requests=workload.requests[i:i + batch_size])
            inputs = preprocess(tokenizer=self.tokenizer, workload=batch, num_tokens=num_tokens_per_request)

            num_tokens.append(count_input_tokens(inputs))
            candidate_losses.append(self.candidate.compute_token_loss(inputs))
            reference_losses.append(self.reference.compute_token_loss(inputs))

        candidate_losses = np.array(candidate_losses)
        reference_losses = np.array(reference_losses)
        num_tokens = np.array(num_tokens)
        total_tokens = num_tokens.sum()

        candidate_loss = (candidate_losses * num_tokens).sum() / total_tokens
        reference_loss = (reference_losses * num_tokens).sum() / total_tokens

        return (candidate_loss - reference_loss) / reference_loss * 100

        
        