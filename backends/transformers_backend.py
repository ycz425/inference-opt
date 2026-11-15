from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from preprocessing import Inputs
from config import Precision, Device
from transformers import BitsAndBytesConfig


DTYPE_MAP: dict[Precision, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}


class TransformersBackend:
    def __init__(self, model_name: str, precision: Precision = 'fp32', output_tokens=256, device: Device ='cuda'):
        if precision in {'int4', 'int8'}:
            quantization_config = BitsAndBytesConfig(load_in_4bit=(precision == 'int4'), load_in_8bit=(precision == 'int8'))
            self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
        else :  
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE_MAP[precision])
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.output_tokens = output_tokens

    def generate(self, inputs: Inputs) -> torch.LongTensor:
        input_ids = torch.tensor(inputs.input_ids, dtype=torch.int64, device=self.model.device)
        attention_mask = torch.tensor(inputs.attention_mask, dtype=torch.int64, device=self.model.device)
        outputs = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            min_new_tokens=self.output_tokens,
            max_new_tokens=self.output_tokens
        )
        return outputs[:, inputs['input_ids'].shape[1]:]

    def detokenize(self, outputs: torch.LongTensor) -> list[str]:
        return self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)

    def synchronize(self) -> None:
        if self.model.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.model.device.type == 'mps':
            torch.mps.synchronize()

    def compute_token_losses(self, inputs: Inputs) -> torch.Tensor:
        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return outputs.logits