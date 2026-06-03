# ============================================================
#  Structured output schema for LLM
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional,List,Literal
from pydantic import BaseModel, Field, model_validator


class Issue(BaseModel):
    """Represents a single issue detected in the text"""
    type: Literal["Violation", "Warning", "Suggestion", "Error", "To Improve"] = Field(default="Error")  # Issue type
    start: int = Field(default=0)  # Start index in text
    end: int = Field(default=0)    # End index in text
    token: str = Field(default="")  # The highlighted word/phrase
    message: str = Field(default="Issue detected")  # Issue description
    suggestion: Optional[str] = None
    is_highlight: bool = True # If True, it is shown as background in the text

    @model_validator(mode='before')
    @classmethod
    def map_hallucinated_fields(cls, data: any) -> any:
        """
        Maps hallucinated fields like 'error_type' to standard fields, 
        and ensures 'type' is one of the allowed Literals.
        """
        if isinstance(data, dict):
            # Map error_type to type or message
            if "error_type" in data and "type" not in data:
                err_val = data.pop("error_type")
                valid_types = ["Violation", "Warning", "Suggestion", "Error", "To Improve"]
                if err_val in valid_types:
                    data["type"] = err_val
                else:
                    data["type"] = "Error"
                    if "message" not in data:
                        data["message"] = f"Type: {err_val}"
            
            # Ensure type is valid if it exists but is wrong
            if "type" in data and data["type"] not in ["Violation", "Warning", "Suggestion", "Error", "To Improve"]:
                if "message" not in data:
                    data["message"] = f"Original type: {data['type']}"
                data["type"] = "Error"
                
        return data

    class Config:
        json_schema_extra = {
            "example": {
                "type": "Error",
                "start": 8,
                "end": 9,
                "token": "e",
                "message": "Use the verb, not the conjunction.",
                "suggestion": "è",
                "is_highlight": True
            }
        }


class TextAnalysisResponse(BaseModel):
    """Complete LLM analysis response"""
    text: Optional[str] = Field(default=None)  # Original analyzed text
    issues: List[Issue] = Field(default_factory=list)  # List of detected issues
    grammar_errors: Optional[bool] = Field(
        default=False,
        description="Does the review contain objective grammatical errors? True (contains errors) or False (is correct)"
    )

    @model_validator(mode='before')
    @classmethod
    def map_errors_to_issues(cls, data: any) -> any:
        """
        Maps 'errors' key to 'issues' if 'issues' is missing.
        Handles LLM hallucinations where 'errors' is used instead of 'issues'.
        """
        if isinstance(data, dict):
            if "errors" in data and "issues" not in data:
                data["issues"] = data.pop("errors")
        return data

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The house is beautiful",
                "issues": [
                    {
                        "type": "Error",
                        "start": 8,
                        "end": 9,
                        "token": "e",
                        "message": "Use the verb, not the conjunction.",
                        "suggestion": "è",
                        "is_highlight": True
                    }
                ],
                "grammar_errors": True
            }
        }

class GuidelinesEvaluation(BaseModel):
    valide: bool = Field(
        description="Does the review respect the input guidelines? True or False"
    )
    title: str = Field(
        description="A short title (max 3-5 words) identifying the PRODUCT or SERVICE reviewed. It MUST NOT be a judgment on the review. Correct example: 'Samsung Galaxy S26' or 'Pizza Restaurant'. Wrong example: 'Valid review' or 'No errors'."
    )
    reasoning: str = Field(
        description="Very brief general analysis of how the review is constructed."
    )


# ============================================================
#  Compliance check result
# ============================================================
@dataclass
class ComplianceResult:
    platform_id: str
    platform_name: str
    review_text: str
    is_generic_compliant: bool
    follow_guidelines: Optional[bool] = None
    grammar_errors: Optional[bool] = None
    title: Optional[str] = None
    reasoning: Optional[str] = None
    highlights: Optional[TextAnalysisResponse] = None
    details: dict = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "platform_id": self.platform_id,
            "platform_name": self.platform_name,
            "is_generic_compliant": self.is_generic_compliant,
            "follow_guidelines": self.follow_guidelines,
            "grammar_errors": self.grammar_errors,
            "title": self.title,
            "reasoning": self.reasoning,
            "insight_score": getattr(self, "insight_score", None),
            "highlights": self.highlights.dict() if self.highlights else None,
            "details": self.details,
        }
