from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from .analyst_response import AnalystResponse
from .validator_response import ValidatorResponse
from .search_result import SearchResult

class WorkflowState(BaseModel):
    query: str
    type_of_query: Optional[str] = None
    sub_queries: List[str] = []
    search_results: Optional[List[SearchResult]] = None
    
    summerized_history: str = ""
    
    output: str = ""
    need_human: bool = False
    
    analysis: Optional[AnalystResponse] = None
    validation: Optional[ValidatorResponse] = None