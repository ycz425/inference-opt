from typing import Protocol, Any
from preprocessing import Inputs
from config import Precision, Device
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import gc


DTYPE_MAP: dict[Precision, str] = {
    "fp32": 'float32',
    "fp16": 'float16',
    "bf16": 'bfloat16'
}


class vLLMBackend:
    def __init__(self, model_name: str, precision: Precision, device: Device, compile: bool = False, batch_size: int = 8):
        if device != 'cuda':
            raise ValueError

        if precision in {'int4', 'int8'}:
            raise ValueError
        
        self.llm = LLM(
            model=model_name,
            dtype=DTYPE_MAP[precision],
            gpu_memory_utilization=0.90,
            enforce_eager=not compile,
            enable_prefix_caching=False,  # so two-step TTFT/TPOT calculation works
            max_num_seqs=batch_size,
            max_num_batched_tokens=batch_size * 1024,  # must be at least batch_size * input_tokens
            skip_tokenizer_init=True,
        )

    def generate(self, inputs: Inputs, output_tokens: int) -> torch.LongTensor:
        prompts = [TokensPrompt(prompt_token_ids=input_id) for input_id in inputs.input_ids]
        outputs = self.llm.generate(
            prompts,
            SamplingParams(
                temperature=0,
                max_tokens=output_tokens,
                min_tokens=output_tokens,
                ignore_eos=True,
                detokenize=False
            ),
            use_tqdm=False
        )
        return torch.tensor([output.outputs[0].token_ids for output in outputs], dtype=torch.long)

    def synchronize(self) -> None:
        # vLLM generate is already blocking
        pass

    def compute_token_loss(self, inputs: Inputs) -> float:
        prompts = [TokensPrompt(prompt_token_ids=input_id) for input_id in inputs.input_ids]
        outputs = self.llm.generate(
            prompts,
            SamplingParams(
                temperature=0,
                max_tokens=1,
                prompt_logprobs=0,
                detokenize=False
            ),
            use_tqdm=False
        )
        logprobs = 0
        n = 0
        for output in outputs:
            for entry in output.prompt_logprobs:
                if entry is not None:
                    logprobs += next(iter(entry.values())).logprob
                    n += 1

        return - logprobs / n

    def free_device_memory(self) -> None:
        destroy_model_parallel()
        destroy_distributed_environment()
        self.llm = None
        gc.collect()
        torch.cuda.empty_cache()

