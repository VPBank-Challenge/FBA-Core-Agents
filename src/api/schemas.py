from pydantic import BaseModel
from typing import Optional, List, Literal

class HistoryMessage(BaseModel):
    role: str = Literal["bot", "user"]
    message: str

class ChatRequest(BaseModel):
    api_key: str
    model: Optional[str] = "gpt-4o-mini"
    question: str
    opensearch_username: str
    opensearch_password: str
    opensearch_endpoint: str
    previous_conversation: Optional[List[HistoryMessage]] = None

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