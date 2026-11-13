from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Literal

type Role = Literal['user', 'assistant']

class Turn(BaseModel):
    role: Role
    content: str

class Conversation(BaseModel):
    id: str
    turns: list[Turn]

class Workload(BaseModel):
    conversations: list[Conversation] 

class WorkloadLoader(ABC):
    @abstractmethod 
    def load_workload(self, size=None, seed=42) -> Workload:
        ...