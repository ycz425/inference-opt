from backends import Backend
from .result import BatchResult
from workload import Workload
from config import Config
from memory import MemoryMonitor
from preprocessing import preprocess
from transformers import AutoTokenizer
import torch
import time


class BenchmarkRunner:
    def __init__(self, backend: Backend, memory_monitor: MemoryMonitor):
        self.backend = backend
        self.memory_monitor = memory_monitor

    def run(self, workload: Workload, config: Config) -> tuple[list[BatchResult], int]:
        results = []
        batch_size = config.batch_size

        if config.warmup_batches >= len(workload.requests) // batch_size:
            raise ValueError

        warmup = config.warmup_batches
        for i in range(0, len(workload.requests), batch_size):
            if warmup == 0:
                self.memory_monitor.reset()

            batch = Workload(requests=workload.requests[i:i + batch_size])

            tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_size='left')
            inputs = preprocess(tokenizer=tokenizer, workload=batch, seed=config.seed)

            input_tokens = torch.tensor(inputs.attention_mask).sum(dtype=torch.int64).item()

            self.backend.synchronize()
            generation_start = time.perf_counter()
            outputs = self.backend.generate(inputs)
            self.backend.synchronize()
            generation_end = time.perf_counter()

            output_texts = self.backend.detokenize(outputs)

            output_tokens = (outputs != tokenizer.pad_token_id).sum(dtype=torch.int64).item()
            generation_time = generation_end - generation_start

            warmup -= 1
            if warmup >= 0:
                continue

            results.append(BatchResult(
                batch_size=len(batch.requests),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=generation_time,
                ids=[request.id for request in batch.requests],
                output_texts=output_texts
            ))

        peak_memory = self.memory_monitor.get_peak_memory()

        return results, peak_memory