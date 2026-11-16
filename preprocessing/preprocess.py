from workload import Workload, Request
from transformers import PreTrainedTokenizerBase
from .inputs import Inputs
from typing import Literal


def to_conversation(request: Request) -> list[dict[str, str]]:
    turns = []
    for turn in request.turns:
        turns.append({'role': turn.role, 'content': turn.content})
    return turns


def preprocess(tokenizer: PreTrainedTokenizerBase, workload: Workload, max_tokens: int | None = None, padding_side: Literal['left', 'right'] = 'left') -> Inputs:
    tokenizer.padding_side = padding_side
    conversations = [to_conversation(request) for request in workload.requests]

    if max_tokens is None:
        batch_encoding = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True
        )
        input_ids = batch_encoding['input_ids']
        attention_mask = batch_encoding['attention_mask']
    else:
        batch_encoding = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True
        )
        input_ids = []
        attention_mask = []
        for i in range(len(batch_encoding['input_ids'])):
            if len(batch_encoding['input_ids'][i]) >= max_tokens:
                input_ids.append(batch_encoding['input_ids'][i][:max_tokens])
                attention_mask.append([1] * max_tokens)
            else:
                length = len(batch_encoding['input_ids'][i])
                if padding_side == 'left':
                    input_ids.append([tokenizer.pad_token_id] * (max_tokens - length) + batch_encoding['input_ids'][i])
                    attention_mask.append([0] * (max_tokens - length) + [1] * length)
                else:
                    input_ids.append(batch_encoding['input_ids'][i] + [tokenizer.pad_token_id] * (max_tokens - length))
                    attention_mask.append([1] * length + [0] * (max_tokens - length))

    return Inputs(input_ids=input_ids, attention_mask=attention_mask)