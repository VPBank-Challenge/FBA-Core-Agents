from pydantic import BaseModel
from typing import List

class ReflectorResponse(BaseModel):
    sub_queries: List[str]