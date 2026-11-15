import torch
from workload import Workload, SyntheticPrompt
from transformers import PreTrainedTokenizerBase
from .inputs import Inputs

def preprocess(tokenizer: PreTrainedTokenizerBase, workload: Workload, seed: int = 42) -> Inputs:
    if all(isinstance(r, SyntheticPrompt) for r in workload.requests):
        n = workload.requests[0].num_tokens
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, tokenizer.vocab_size, (len(workload.requests), n), generator=g)
        return Inputs(input_ids=ids, attention_mask=torch.ones_like(ids))
    
    conversations = []
    for conversation in workload.requests:
        turns = []
        for turn in conversation.turns:
            turns.append({'role': turn.role, 'content': turn.content})
        conversations.append(turns)
    batch_encoding = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_dict=True
    )

    return Inputs(input_ids=batch_encoding['input_ids'], attention_mask=batch_encoding['attention_mask'])