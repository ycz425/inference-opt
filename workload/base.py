from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Literal

Role = Literal['user', 'assistant']

class Turn(BaseModel):
    role: Role
    content: str

class Request(BaseModel):
    id: str
    turns: list[Turn]

    def to_conversation(self) -> list[dict[str, str]]:
        turns = []
        for turn in self.turns:
            turns.append({'role': turn.role, 'content': turn.content})
        return turns

class Workload(BaseModel):
    requests: list[Request] 

class WorkloadLoader(ABC):
    @abstractmethod 
    def load_workload(self, num_requests: int, min_tokens: int | None = None, tokenizer=None, seed=42) -> Workload:
        ...