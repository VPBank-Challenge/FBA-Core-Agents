from typing import Literal
from pydantic import BaseModel

class ReceptionistResponse(BaseModel):
    type_of_query: Literal["small_talk", "out_of_scope", "banking"]
    content: str