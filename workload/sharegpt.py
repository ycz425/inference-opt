from .base import WorkloadLoader, Workload, Turn, Request
from datasets import load_dataset


class ShareGPTLoader(WorkloadLoader):
    def load_workload(self, num_requests, seed=42) -> Workload:
        dataset = load_dataset('RyokoAI/ShareGPT52K', split='train')
        dataset = dataset.shuffle(seed=seed).select(range(num_requests))

        conversations = []
        for row in dataset:
            row_id = row['id']
            conversation = row['conversations']
            turns = []
            for i, turn in enumerate(conversation):
                if turn['from'] == 'human':
                    turns.append(Turn(role='user', content=turn['value']))
                elif turn['from'] == 'gpt' and i < len(conversation) - 1:
                    turns.append(Turn(role='assistant', content=turn['value']))
                elif turn['from'] == 'gpt':
                    break
                else:
                    raise ValueError
            conversations.append(Request(id=row_id, turns=turns))
        return Workload(requests=conversations)

        