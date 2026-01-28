from pydantic import BaseModel
from typing import List, Optional, Any


class AnswerOutput(BaseModel):
    answer: str
    pages: List[int]

    docs: Optional[List[Any]] = None
