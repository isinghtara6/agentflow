from typing import Literal

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    style: Literal["concise", "detailed", "bullets"] = "concise"


class RunResponse(BaseModel):
    request_id: str
    answer: str
    subtasks: list[str]
    review_rounds: int
    tokens_used: int
    cached: bool
    redactions: list[str]
    latency_ms: int
