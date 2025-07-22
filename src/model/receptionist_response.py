from pydantic import BaseModel
from typing import List, Optional, Literal

class ReceptionistResponse(BaseModel):
    type_of_query: Literal["social", "banking", "out_of_scope"]
    content: str
    
    main_topic: str
    key_information: List[str]
    clarified_query: str
    customer_type: Literal["Individual", "Micro Business", "SME", "Large Enterprise"]