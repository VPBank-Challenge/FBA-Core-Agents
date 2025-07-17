from pydantic import BaseModel
from typing import Optional

class SearchResult(BaseModel):
    content: str
    citation: Optional[str]