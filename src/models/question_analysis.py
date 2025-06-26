from pydantic import BaseModel
from typing import List, Optional


class QuestionAnalysis(BaseModel):
    main_topic: str
    key_information: List[str]
    clarified_query: str
    customer_type: Optional[str] = None  # Individual, Micro Business, SME, Large Enterprise
    