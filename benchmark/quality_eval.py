from backends import Backend
from workload import Workload
from preprocessing import preprocess
from workload import workload_from_config
from transformers import PreTrainedTokenizerBase
from config import QualityConfig
from utils import count_input_tokens
import numpy as np

class QualityEvaluator:
    def __init__(self, candidate: Backend, reference: Backend, tokenizer: PreTrainedTokenizerBase):
        self.candidate = candidate
        self.reference = reference
        self.tokenizer = tokenizer

    def run(self, config: QualityConfig) -> float:
        candidate_losses = []
        reference_losses = []
        num_tokens = []

        workload = workload_from_config(config.workload, self.tokenizer)
        batch_size = config.reference.batch_size

        for i in range(0, len(workload.requests), batch_size):
            batch = Workload(requests=workload.requests[i:i + batch_size])
            inputs = preprocess(tokenizer=self.tokenizer, workload=batch, max_tokens=config.workload.num_tokens, padding_side='right')

            num_tokens.append(count_input_tokens(inputs) - len(batch.requests))  # ignoring first token of each request
            candidate_losses.append(self.candidate.compute_token_loss(inputs))
            reference_losses.append(self.reference.compute_token_loss(inputs))

        candidate_losses = np.array(candidate_losses)
        reference_losses = np.array(reference_losses)
        num_tokens = np.array(num_tokens)
        total_tokens = num_tokens.sum()

        candidate_loss = (candidate_losses * num_tokens).sum() / total_tokens
        reference_loss = (reference_losses * num_tokens).sum() / total_tokens

        return (candidate_loss - reference_loss) / reference_loss * 100

        
        