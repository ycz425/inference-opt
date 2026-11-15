from pydantic import BaseModel


class Inputs(BaseModel):
    input_ids: list[list[int]]
    attention_mask: list[list[int]]