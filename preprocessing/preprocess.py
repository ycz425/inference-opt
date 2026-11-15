from workload import Workload
from transformers import PreTrainedTokenizerBase
from .inputs import Inputs
from typing import Literal

def preprocess(tokenizer: PreTrainedTokenizerBase, workload: Workload, num_tokens: int | None = None, padding_side: Literal['left', 'right'] = 'left') -> Inputs:
    tokenizer.padding_side = padding_side
    conversations = []
    for conversation in workload.requests:
        turns = []
        for turn in conversation.turns:
            turns.append({'role': turn.role, 'content': turn.content})
        conversations.append(turns)

    if num_tokens is None:
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
            if len(batch_encoding['input_ids'][i]) >= num_tokens:
                input_ids.append(batch_encoding['input_ids'][i][:num_tokens])
                attention_mask.append([1] * num_tokens)
            else:
                length = len(batch_encoding['input_ids'][i])
                if padding_side == 'left':
                    input_ids.append([tokenizer.pad_token_id] * (num_tokens - length) + batch_encoding['input_ids'][i])
                    attention_mask.append([0] * (num_tokens - length) + [1] * length)
                else:
                    input_ids.append(batch_encoding['input_ids'][i] + [tokenizer.pad_token_id] * (num_tokens - length))
                    attention_mask.append([1] * length + [0] * (num_tokens - length))

    return Inputs(input_ids=input_ids, attention_mask=attention_mask)