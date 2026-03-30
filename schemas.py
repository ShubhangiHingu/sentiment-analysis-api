from pydantic import BaseModel, Field

class AnalysisRequest(BaseModel):
    # Reject empty strings using min_length
    text: str = Field(..., min_length=1, description="The text to analyze")

class AnalysisResponse(BaseModel):
    label: str
    score: float