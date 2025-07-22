# src/api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class HistoryMessage(BaseModel):
    role: Literal["user", "bot"]
    message: str

class ChatRequest(BaseModel):
    question: str = Field(..., description="User's question")
    previous_conversation: List[HistoryMessage] = Field(default=[], description="Conversation history")

class ChatResponse(BaseModel):
    question: str
    answer: str
    main_topic: str
    key_information: List[str]
    clarified_query: str
    customer_type: str
    type_of_query: str
    need_human: bool
    confidence_score: float
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

class SetupRequest(BaseModel):
    api_key: str
    model: str
    opensearch_endpoint: str
    opensearch_username: str
    opensearch_password: str