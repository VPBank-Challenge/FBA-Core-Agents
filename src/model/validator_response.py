from pydantic import BaseModel
from typing import Literal

class ValidatorResponse(BaseModel):
    verdict: Literal["sufficient", "insufficient"]
    rationale: str