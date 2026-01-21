from pydantic import BaseModel, Field
from typing import List

class AnswerOutput(BaseModel):
    answer: str = Field(
        description="Final answer generated from the document context"
    )

    pages: List[int] = Field(
        description="Page numbers used to generate the answer"
    )
