from pydantic import BaseModel

class BatchResult(BaseModel):
    batch_size: int
    input_tokens: int
    output_tokens: int
    generation_time: float
    ids: list[str]
    output_texts: list[str]


class BenchmarkResult(BaseModel):
    num_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_generation_time: float

    output_tokens_per_sec: float
    total_tokens_per_sec: float
    requests_per_sec: float

    average_batch_latency: float

    peak_memory: int

    batch_results: list[BatchResult]


def benchmark_result(batch_results: list[BatchResult], peak_memory: int) -> BenchmarkResult:
    num_requests = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_generation_time = 0

    for result in batch_results:
        num_requests += result.batch_size
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens
        total_generation_time += result.generation_time

    output_tokens_per_sec = total_output_tokens / total_generation_time
    total_tokens_per_sec = (total_input_tokens + total_output_tokens) / total_generation_time
    requests_per_sec = num_requests / total_generation_time

    average_batch_latency = total_generation_time / len(batch_results)

    return BenchmarkResult(
        num_requests=num_requests,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_generation_time=total_generation_time,
        output_tokens_per_sec=output_tokens_per_sec,
        total_tokens_per_sec=total_tokens_per_sec,
        requests_per_sec=requests_per_sec,
        average_batch_latency=average_batch_latency,
        peak_memory=peak_memory,
        batch_results=batch_results
    )