from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ReceptionistResponse(BaseModel):
    type_of_query: int # 0: Small Talk, 2: Out of Scope, 1: Banking Query
    content: str