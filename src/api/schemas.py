from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class HistoryMessage(BaseModel):
    role: str = Literal["bot", "user"]
    message: str = Field(..., description="Message from the user or bot in the conversation history")

class ChatRequest(BaseModel):
    api_key: str = Field(..., description="LLM API key required for authentication")
    model: Optional[str] = Field(default="gpt-4o-mini", description="LLM model to use")
    question: str = Field(..., description="User's current question")
    opensearch_username: str = Field(..., description="OpenSearch username")
    opensearch_password: str = Field(..., description="OpenSearch password")
    opensearch_endpoint: str = Field(..., description="OpenSearch endpoint URL")
    previous_conversation: Optional[List[HistoryMessage]] = Field(
        default_factory=List,
        description="Previous Conversation"
    )

class ChatResponse(BaseModel):
    question: str = Field(..., description="Original question from the user")
    answer: str = Field(..., description="Model's answer to the user question")
    main_topic: str = Field(..., description="Main detected topic")
    key_information: str = Field(..., description="Extracted key information from the query")
    clarified_query: str = Field(..., description="Refined or clarified query after analysis")
    customer_type: str = Field(..., description="Customer type classification")
    type_of_query: str = Field(..., description="Query type classification")
    need_human: bool = Field(..., description="Whether a human agent is required")
    confidence_score: float = Field(..., description="Confidence score of the answer")
