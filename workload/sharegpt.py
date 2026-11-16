from .base import WorkloadLoader, Workload, Turn, Request
from datasets import load_dataset


class ShareGPTLoader(WorkloadLoader):
    def load_workload(self, num_requests: int, min_tokens: int | None = None, tokenizer=None, seed=42) -> Workload:
        if min_tokens is not None and tokenizer is None:
            raise ValueError

        dataset = load_dataset('RyokoAI/ShareGPT52K', split='train')
        dataset = dataset.shuffle(seed=seed)

        conversations = []
        for row in dataset:
            row_id = row['id']
            conversation = row['conversations']
            turns = []
            valid = True
            for i, turn in enumerate(conversation):
                if turn['from'] == 'human':
                    turns.append(Turn(role='user', content=turn['value']))
                elif turn['from'] == 'gpt' and i < len(conversation) - 1:
                    turns.append(Turn(role='assistant', content=turn['value']))
                elif turn['from'] == 'gpt':
                    break
                else:
                    valid = False
                    break
            if not valid:
                continue

            request = Request(id=row_id, turns=turns)
            
            if min_tokens is not None:
                conversation = request.to_conversation()
                tokens = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    padding=False,
                    return_dict=False
                )
                if len(tokens) < min_tokens:
                    continue
            
            conversations.append(Request(id=row_id, turns=turns))

            if len(conversations) == num_requests:
                break

        return Workload(requests=conversations)

        