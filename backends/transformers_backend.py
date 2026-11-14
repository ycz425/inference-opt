from transformers import BatchEncoding, AutoModelForCausalLM, AutoTokenizer
import torch
from workloads import Workload, SyntheticPrompt, Conversation
from config import Precision, Device


DTYPE_MAP: dict[Precision, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class TransformersBackend:
    def __init__(self, model_name: str, precision: Precision = 'fp32', output_tokens=256, device: Device ='cuda', seed=42):
        self.model = AutoModelForCausalLM.from_pretrained(model_name,dtype=DTYPE_MAP[precision]).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.output_tokens = output_tokens
        self.generator = torch.Generator().manual_seed(seed) 

    def preprocess(self, workload: Workload) -> BatchEncoding:
        if all(isinstance(r, SyntheticPrompt) for r in workload.requests):
            n = workload.requests[0].num_tokens
            ids = torch.randint(0, self.tokenizer.vocab_size, (len(workload.requests), n), generator=self.generator)
            return BatchEncoding({'input_ids': ids, 'attention_mask': torch.ones_like(ids)})
        
        conversations = []
        for conversation in workload.requests:
            turns = []
            for turn in conversation.turns:
                turns.append({'role': turn.role, 'content': turn.content})
            conversations.append(turns)
        return self.tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_tensors='pt'
        )
        

    def generate(self, inputs: BatchEncoding) -> torch.LongTensor:
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(
            inputs=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            do_sample=False,
            min_new_tokens=self.output_tokens,
            max_new_tokens=self.output_tokens
        )
        return outputs[:, inputs['input_ids'].shape[1]:]

    def detokenize(self, outputs: torch.LongTensor) -> list[str]:
        return self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)

    def count_input_tokens(self, inputs: BatchEncoding) -> int:
        return inputs['attention_mask'].sum(dtype=torch.int64).item()

    def count_output_tokens(self, outputs: torch.LongTensor) -> int:
        return (outputs != self.tokenizer.pad_token_id).sum(dtype=torch.int64).item()

    def synchronize(self) -> None:
        if self.model.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.model.device.type == 'mps':
            torch.mps.synchronize()
