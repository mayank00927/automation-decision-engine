from pydantic import BaseModel, Field
from typing import Literal


class Tool(BaseModel):
    name: str
    purpose: str
    category: Literal["Trigger", "Action", "AI", "Storage", "Notification"]


class AutomationResponse(BaseModel):
    problem: str = Field(..., description="The original automation idea or problem statement")
    automation_type: str = Field(..., description="Type of automation, e.g. RPA, AI-assisted, rule-based")
    decision: Literal["Automate It", "Needs Review", "Skip It"] = Field(
        ..., description="Top-level recommendation verdict"
    )
    tools: list[Tool] = Field(default_factory=list, description="Recommended tools for the workflow")
    llm: str = Field(..., description="Suggested LLM model, e.g. 'GPT-4o mini', 'Claude Haiku', or 'none'")
    workflow: list[str] = Field(default_factory=list, description="Ordered steps describing the automation flow")
    cost_estimate: str = Field(..., description="Estimated monthly tool cost, e.g. '$20-$50/month'")
    roi: str = Field(..., description="ROI score, e.g. 'Excellent', 'Good', 'Marginal', 'Poor'")
    risks: list[str] = Field(default_factory=list, description="Potential risks or caveats to be aware of")
    mvp_plan: list[str] = Field(default_factory=list, description="Step-by-step plan to build the minimum viable automation")
