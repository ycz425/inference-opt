from workload import Workload, Request
from transformers import PreTrainedTokenizerBase
from .inputs import Inputs
from typing import Literal
from config import BackendType


def preprocess(backend: BackendType, tokenizer: PreTrainedTokenizerBase, workload: Workload, max_tokens: int | None = None, padding_side: Literal['left', 'right'] = 'left') -> Inputs:
    tokenizer.padding_side = padding_side
    conversations = [request.to_conversation() for request in workload.requests]

    batch_encoding = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        padding=(backend != 'vllm' and max_tokens is None),  # if min_tokens = max_tokens, then all inputs are guaranteed to have the same length
        return_dict=True
    )

    if max_tokens is None:
        input_ids = batch_encoding['input_ids']
        attention_mask = batch_encoding['attention_mask']
    else:
        input_ids = []
        attention_mask = []
        for i in range(len(conversations)):
            input_ids.append(batch_encoding['input_ids'][i][:max_tokens])
            attention_mask.append(batch_encoding['attention_mask'][i][:max_tokens])

    return Inputs(input_ids=input_ids, attention_mask=attention_mask)