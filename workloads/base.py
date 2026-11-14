from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Literal

type Role = Literal['user', 'assistant']
type WorkloadType = Literal['synthetic', 'conversation']

class Turn(BaseModel):
    role: Role
    content: str

class Request(BaseModel):
    id: str

class Conversation(Request):
    turns: list[Turn]

class SyntheticPrompt(Request):
    num_tokens: int

class Workload(BaseModel):
    requests: list[Request] 

class WorkloadLoader(ABC):
    @abstractmethod 
    def load_workload(self, size, input_tokens=None, seed=42) -> Workload:
        ...