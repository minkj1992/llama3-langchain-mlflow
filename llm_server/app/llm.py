from langchain_core.pydantic_v1 import BaseModel, Field


class CustomLLMResult(BaseModel):
    a_score: int = Field(description="Score of Team A")
    a_reason: str = Field(description="Reason for deduction for Team A")
    b_score: int = Field(description="Score of Team B")
    b_reason: str = Field(description="Reason for deduction for Team B")
