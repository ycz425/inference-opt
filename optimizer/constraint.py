from benchmark import BenchmarkResult
from pydantic import BaseModel


class Constraint(BaseModel):
    max_memory: int | None
    min_output_tokens_per_sec: float | None
    max_average_batch_latency: float | None


def constraint_satisfied(result: BenchmarkResult, constraint: Constraint | None):
    return (
        constraint is None or
        (
            (constraint.min_output_tokens_per_sec is None or result.output_tokens_per_sec >= constraint.min_output_tokens_per_sec) and
            (constraint.max_memory is None or result.peak_memory <= constraint.max_memory) and
            (constraint.max_average_batch_latency is None or result.average_batch_latency <= constraint.max_average_batch_latency)
        )
    )