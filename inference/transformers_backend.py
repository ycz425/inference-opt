from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase
import torch
from workloads import Workload


class TransformersBackend:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device='cuda', max_new_tokens=1024):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def preprocess(self, workload: Workload) -> BatchEncoding:
        conversations = []
        for conversation in workload.conversations:
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
            max_new_tokens=self.max_new_tokens,
        )
        return outputs[:, inputs['input_ids'].shape[1]:]

    def detokenize(self, outputs: torch.LongTensor) -> list[str]:
        return self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)

    def count_input_tokens(self, inputs: BatchEncoding) -> int:
        return inputs['attention_mask'].sum(dtype=torch.int64).item()

    def count_output_tokens(self, outputs: torch.LongTensor) -> int:
        return (outputs != self.tokenizer.pad_token_id).sum(dtype=torch.int64).item()