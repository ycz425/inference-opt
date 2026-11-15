from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Literal

type Role = Literal['user', 'assistant']

class Turn(BaseModel):
    role: Role
    content: str

class Request(BaseModel):
    id: str
    turns: list[Turn]

class Workload(BaseModel):
    requests: list[Request] 

class WorkloadLoader(ABC):
    @abstractmethod 
    def load_workload(self, num_requests, seed=42) -> Workload:
        ...