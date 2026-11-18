from pydantic import BaseModel
import numpy as np

class BatchResult(BaseModel):
    batch_size: int
    input_tokens: int
    output_tokens: int
    generation_time: float
    ttft: float
    tpot: float
    ids: list[str]


class BenchmarkResult(BaseModel):
    num_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_generation_time: float

    output_tokens_per_sec: float
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float
    requests_per_sec: float

    mean_ttft: float
    median_ttft: float

    mean_tpot: float
    median_tpot: float

    mean_e2e_latency: float

    model_memory: int | None
    peak_memory: int | None
    runtime_memory: int | None

    ce_degradation: float | None

    batch_results: list[BatchResult]


def benchmark_result(batch_results: list[BatchResult], peak_memory: int | None, model_memory: int | None, ce_degradation: float | None) -> BenchmarkResult:
    num_requests = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_generation_time = 0

    ttfts = []
    tpots = []

    for result in batch_results:
        num_requests += result.batch_size
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens
        total_generation_time += result.generation_time
        ttfts.append(result.ttft)
        tpots.append(result.tpot)

    ttfts = np.array(ttfts)
    tpots = np.array(tpots)

    total_ttft = ttfts.sum()
    output_tokens_per_sec = total_output_tokens / total_generation_time
    prefill_tokens_per_sec = total_input_tokens / total_ttft
    decode_tokens_per_sec = (total_output_tokens - num_requests) / (total_generation_time - total_ttft)
    requests_per_sec = num_requests / total_generation_time

    mean_ttft = ttfts.mean()
    median_ttft = np.median(ttfts)
    mean_tpot = tpots.mean()
    median_tpot = np.median(tpots)
    
    mean_e2e_latency = total_generation_time / len(batch_results)

    if peak_memory is None or model_memory is None:
        runtime_memory = None
    else:
        runtime_memory = peak_memory - model_memory

    return BenchmarkResult(
        num_requests=num_requests,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_generation_time=total_generation_time,
        output_tokens_per_sec=output_tokens_per_sec,
        decode_tokens_per_sec=decode_tokens_per_sec,
        prefill_tokens_per_sec=prefill_tokens_per_sec,
        requests_per_sec=requests_per_sec,
        mean_ttft=mean_ttft,
        median_ttft=median_ttft,
        mean_tpot=mean_tpot,
        median_tpot=median_tpot,

        mean_e2e_latency=mean_e2e_latency,

        model_memory=model_memory,
        peak_memory=peak_memory,
        runtime_memory=runtime_memory,

        ce_degradation=ce_degradation,
        batch_results=batch_results
    )