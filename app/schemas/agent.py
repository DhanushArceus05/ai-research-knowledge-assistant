from typing import Optional, List, Any, Dict
from pydantic import BaseModel, field_validator


class AgentRunRequest(BaseModel):
    instruction: str
    session_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    top_k: Optional[int] = None

    @field_validator("instruction")
    @classmethod
    def instruction_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("instruction must not be empty")
        return v


class AgentTraceStep(BaseModel):
    step: int
    action: str
    status: str


class AgentRunResponse(BaseModel):
    success: bool
    selected_tool: Optional[str] = None
    trace: List[AgentTraceStep]
    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = []
    session_id: Optional[str] = None
    error: Optional[str] = None
