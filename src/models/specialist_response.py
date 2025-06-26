from pydantic import BaseModel

class SpecialistResponse(BaseModel):
    output: str
    need_human: bool