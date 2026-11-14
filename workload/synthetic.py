from .base import WorkloadLoader, Workload, SyntheticPrompt

class SyntheticLoader(WorkloadLoader):
    def load_workload(self, size, input_tokens=None, seed=42):
        if input_tokens is None:
            raise ValueError

        requests = []
        for i in range(size):
            requests.append(SyntheticPrompt(id=f'syn-{i}', num_tokens=input_tokens))
        
        return Workload(requests=requests)