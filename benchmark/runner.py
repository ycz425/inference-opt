from backends import Backend, backend_from_config
from .result import BatchResult
from workload import Workload, workload_from_config
from config import Config
from memory import MemoryMonitor
from preprocessing import preprocess
from transformers import AutoTokenizer
from .quality_eval import QualityEvaluator
from utils import count_input_tokens, count_output_tokens
import time


class BenchmarkRunner:
    def __init__(self, backend: Backend, memory_monitor: MemoryMonitor):
        self.backend = backend
        self.memory_monitor = memory_monitor

    def run(self, workload: Workload, config: Config) -> tuple[list[BatchResult], int, float]:
        results = []
        batch_size = config.inference_config.batch_size

        tokenizer = AutoTokenizer.from_pretrained(config.inference_config.model_name)

        if config.warmup_batches >= len(workload.requests) // batch_size:
            raise ValueError

        warmup = config.warmup_batches
        for i in range(0, len(workload.requests), batch_size):
            if warmup == 0:
                self.memory_monitor.reset()

            batch = Workload(requests=workload.requests[i:i + batch_size])

            inputs = preprocess(tokenizer=tokenizer, workload=batch, num_tokens=config.input_tokens, padding_side='left')

            input_tokens = count_input_tokens(inputs)

            self.backend.synchronize()
            generation_start = time.perf_counter()
            outputs = self.backend.generate(inputs, config.output_tokens)
            self.backend.synchronize()
            generation_end = time.perf_counter()

            output_texts = self.backend.detokenize(outputs)

            output_tokens = count_output_tokens(outputs, tokenizer.pad_token_id)
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

        if config.quality_config is not None:
            workload = workload_from_config(config.quality_config.workload)
            reference = backend_from_config(config.quality_config.reference)
            evaluator = QualityEvaluator(candidate=self.backend, reference=reference, tokenizer=tokenizer)
            ce_degradation = evaluator.run(workload, config.quality_config.num_tokens_per_request, config.quality_config.reference.batch_size)
        else:
            ce_degradation = None

        return results, peak_memory, ce_degradation