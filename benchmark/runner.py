from backends import Backend, backend_from_config
from .result import BatchResult
from workload import Workload
from config import Config
from memory import MemoryMonitor
from preprocessing import preprocess
from transformers import PreTrainedTokenizerBase
from .quality_eval import QualityEvaluator
from utils import count_input_tokens, count_output_tokens
import time


class BenchmarkRunner:
    def __init__(self, backend: Backend, memory_monitor: MemoryMonitor):
        self.backend = backend
        self.memory_monitor = memory_monitor

    def run(self, workload: Workload, config: Config, tokenizer: PreTrainedTokenizerBase) -> tuple[list[BatchResult], int, float | None]:
        results = []
        num_requests = len(workload.requests)
        batch_size = config.inference_config.batch_size

        if num_requests % batch_size != 0:
            raise ValueError

        # WARM UP
        for _ in range(config.inference_config.warmup_batches):
            batch = Workload(requests=workload.requests[0:batch_size])
            inputs = preprocess(tokenizer=tokenizer, workload=batch, max_tokens=config.workload_config.num_tokens, padding_side='left')
            self.backend.generate(inputs, output_tokens=1)
            self.backend.generate(inputs, output_tokens=config.inference_config.output_tokens)

        self.backend.synchronize()
        self.memory_monitor.reset()
        for i in range(0, len(workload.requests), batch_size):
            batch = Workload(requests=workload.requests[i:i + batch_size])

            inputs = preprocess(tokenizer=tokenizer, workload=batch, max_tokens=config.workload_config.num_tokens, padding_side='left')

            input_tokens = count_input_tokens(inputs)

            self.backend.synchronize()
            prefill_start = time.perf_counter()
            self.backend.generate(inputs, output_tokens=1)
            self.backend.synchronize()
            first_token = time.perf_counter()

            ttft = first_token - prefill_start

            self.backend.synchronize()
            generation_start = time.perf_counter()
            outputs = self.backend.generate(inputs, output_tokens=config.inference_config.output_tokens)
            self.backend.synchronize()
            generation_end = time.perf_counter()

            output_texts = self.backend.detokenize(outputs)

            output_tokens = count_output_tokens(outputs, tokenizer.pad_token_id)
            generation_time = generation_end - generation_start

            tpot = (generation_time - ttft) / (config.inference_config.output_tokens - 1)

            results.append(BatchResult(
                batch_size=len(batch.requests),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=generation_time,
                ttft=ttft,
                tpot=tpot,
                ids=[request.id for request in batch.requests],
                output_texts=output_texts
            ))

        peak_memory = self.memory_monitor.get_peak_memory()

        if config.quality_config is not None:
            reference = backend_from_config(config.quality_config.reference)
            evaluator = QualityEvaluator(candidate=self.backend, reference=reference, tokenizer=tokenizer)
            ce_degradation = evaluator.run(config.quality_config)
        else:
            ce_degradation = None

        return results, peak_memory, ce_degradation