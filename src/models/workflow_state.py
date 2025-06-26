from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from .analyst_response import AnalystResponse

class WorkflowState(BaseModel):
    query: str
    type_of_query: Optional[int] = None # 0: Small Talk, 2: Out of Scope, 1: Banking Query
    search_results: str = ""
    output: str = ""
    need_human: bool = False
    analysis: Optional[AnalystResponse] = None