from inference import Backend
from benchmark import BatchResult
from workloads import Workload
import time


class BenchmarkRunner:
    def __init__(self, backend: Backend):
        self.backend = backend

    def run(self, workload: Workload, batch_size=1) -> list[BatchResult]:
        results = []
        for i in range(0, len(workload.conversations), batch_size):
            batch = Workload(conversations=workload.conversations[i:i + batch_size])
            
            inputs = self.backend.preprocess(batch)

            input_tokens = self.backend.count_input_tokens(inputs)

            generation_start = time.perf_counter()
            outputs = self.backend.generate(inputs)
            generation_end = time.perf_counter()

            output_texts = self.backend.detokenize(outputs)

            output_tokens = self.backend.count_output_tokens(outputs)
            generation_time = generation_end - generation_start

            results.append(BatchResult(
                batch_size=len(batch.conversations),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=generation_time,
                ids=[conv.id for conv in batch.conversations],
                output_texts=output_texts
            ))

        return results