from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from preprocessing import Inputs
from config import Precision, Device
from transformers import BitsAndBytesConfig, CompileConfig
import gc

from transformers import logging
logging.disable_progress_bar()


DTYPE_MAP: dict[Precision, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}


class TransformersBackend:
    def __init__(self, model_name: str, precision: Precision = 'fp32', device: Device ='cuda', compile: bool = False, batch_size: int = 8):
        if precision in {'int4', 'int8'}:
            if device != 'cuda':
                raise ValueError
            quantization_config = BitsAndBytesConfig(load_in_4bit=(precision == 'int4'), load_in_8bit=(precision == 'int8'))
            self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
        else :  
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE_MAP[precision]).to(device)

        if compile:
            torch._dynamo.reset()
            self.model.generation_config.cache_implementation = 'static'
            self.model.generation_config.compile_config = CompileConfig(fullgraph=True, mode="reduce-overhead")
        
        self.device = device

    def generate(self, inputs: Inputs, output_tokens: int = 256) -> torch.LongTensor:
        input_ids = torch.tensor(inputs.input_ids, dtype=torch.int64, device=self.model.device)
        attention_mask = torch.tensor(inputs.attention_mask, dtype=torch.int64, device=self.model.device)
        outputs = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            min_new_tokens=output_tokens,
            max_new_tokens=output_tokens
        )
        return outputs[:, input_ids.shape[1]:]

    def synchronize(self) -> None:
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elif self.device == 'mps':
            torch.mps.synchronize()

    def compute_token_loss(self, inputs: Inputs) -> float:
        input_ids = torch.tensor(inputs.input_ids, dtype=torch.int64, device=self.model.device)
        attention_mask = torch.tensor(inputs.attention_mask, dtype=torch.int64, device=self.model.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=torch.where(attention_mask == 0, -100, input_ids))
        return outputs.loss.item()

    def free_device_memory(self):
        self.model = None
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()