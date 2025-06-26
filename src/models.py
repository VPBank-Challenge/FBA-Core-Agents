from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class QuestionAnalysis(BaseModel):
    main_topic: str
    key_information: List[str]
    clarified_query: str
    customer_type: Optional[str] = None  # Individual, Micro Business, SME, Large Enterprise

class ReceptionistResponse(BaseModel):
    type_of_query: int # 0: Small Talk, 2: Out of Scope, 1: Banking Query
    content: str

class ResearchState(BaseModel):
    query: str
    search_results: str = ""
    output: str = ""
    analysis: Optional[QuestionAnalysis] = None
    should_end: bool = False