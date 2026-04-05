"""
workflow_builder.py — Converts a raw LLM JSON response into a structured
five-stage workflow: Trigger → Input Collection → Processing → Decision → Output.
"""

from dataclasses import dataclass, field
from typing import Any

# ── Stage keywords used to classify free-text workflow steps ─────────────────
_STAGE_KEYWORDS: dict[str, list[str]] = {
    "trigger": [
        "trigger", "start", "receive", "webhook", "schedule", "cron",
        "event", "watch", "monitor", "detect", "listen", "on submit",
        "new row", "incoming", "arrival",
    ],
    "input_collection": [
        "collect", "fetch", "read", "pull", "extract", "gather", "import",
        "load", "retrieve", "get data", "form", "input", "capture", "ingest",
        "query", "scrape",
    ],
    "processing": [
        "process", "transform", "parse", "enrich", "classify", "summarise",
        "summarize", "analyse", "analyze", "filter", "clean", "format",
        "convert", "map", "calculate", "generate", "run", "execute", "call api",
        "llm", "ai", "model", "embed", "chunk",
    ],
    "decision": [
        "decide", "decision", "check", "validate", "verify", "route", "branch",
        "condition", "if ", "approve", "reject", "evaluate", "assess", "score",
        "threshold", "match", "compare",
    ],
    "output": [
        "send", "notify", "post", "write", "save", "store", "update", "insert",
        "create", "email", "slack", "webhook", "export", "publish", "respond",
        "reply", "output", "return", "log", "record", "display",
    ],
}

# Tool category → workflow stage mapping
_CATEGORY_TO_STAGE: dict[str, str] = {
    "Trigger":      "trigger",
    "Action":       "output",
    "AI":           "processing",
    "Storage":      "output",
    "Notification": "output",
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class WorkflowStage:
    name: str                        # human-readable stage name
    steps: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"stage": self.name, "steps": self.steps, "tools": self.tools}


@dataclass
class StructuredWorkflow:
    trigger:          WorkflowStage
    input_collection: WorkflowStage
    processing:       WorkflowStage
    decision:         WorkflowStage
    output:           WorkflowStage

    # metadata passed through from the LLM response
    verdict:    str = ""
    confidence: str = ""
    reason:     str = ""

    def stages(self) -> list[WorkflowStage]:
        return [
            self.trigger,
            self.input_collection,
            self.processing,
            self.decision,
            self.output,
        ]

    def as_dict(self) -> dict:
        return {
            "verdict":    self.verdict,
            "confidence": self.confidence,
            "reason":     self.reason,
            "workflow": [s.as_dict() for s in self.stages()],
        }

    def summary(self) -> str:
        lines = [
            f"Decision : {self.verdict} ({self.confidence})",
            f"Reason   : {self.reason}",
            "",
        ]
        for stage in self.stages():
            lines.append(f"[{stage.name}]")
            for step in stage.steps:
                lines.append(f"  - {step}")
            if stage.tools:
                lines.append(f"  Tools: {', '.join(stage.tools)}")
            lines.append("")
        return "\n".join(lines).rstrip()


# ── Core function ─────────────────────────────────────────────────────────────

def build_workflow(llm_response: dict[str, Any]) -> StructuredWorkflow:
    """
    Convert a raw LLM JSON response (as returned by LLMService.analyze) into
    a StructuredWorkflow with five labelled stages.

    Parameters
    ----------
    llm_response : dict
        The parsed JSON dict from the LLM. Expected keys (all optional but
        handled gracefully if missing):
            - "workflow"         : list[str]  — free-text steps
            - "tools"            : list[dict] — {name, purpose, category}
            - "decision"         : dict       — {verdict, confidence, reason}
            - "llm_recommendation": dict      — {use_llm, model_suggestion, ...}

    Returns
    -------
    StructuredWorkflow
    """
    stages = _empty_stages()

    _classify_workflow_steps(llm_response.get("workflow", []), stages)
    _assign_tools(llm_response.get("tools", []), stages)
    _inject_llm_step(llm_response.get("llm_recommendation", {}), stages)
    _fill_empty_stages(stages)

    decision = llm_response.get("decision", {})
    return StructuredWorkflow(
        trigger=stages["trigger"],
        input_collection=stages["input_collection"],
        processing=stages["processing"],
        decision=stages["decision"],
        output=stages["output"],
        verdict=decision.get("verdict", ""),
        confidence=decision.get("confidence", ""),
        reason=decision.get("reason", ""),
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _empty_stages() -> dict[str, WorkflowStage]:
    return {
        "trigger":          WorkflowStage("Trigger"),
        "input_collection": WorkflowStage("Input Collection"),
        "processing":       WorkflowStage("Processing"),
        "decision":         WorkflowStage("Decision"),
        "output":           WorkflowStage("Output"),
    }


def _classify_workflow_steps(
    steps: list[str],
    stages: dict[str, WorkflowStage],
) -> None:
    """
    Assign each free-text workflow step to the best-matching stage by keyword
    scanning. Falls back to 'processing' when nothing matches.
    """
    for step in steps:
        lower = step.lower()
        matched = _best_stage(lower)
        stages[matched].steps.append(step)


def _best_stage(text: str) -> str:
    """Return the stage key whose keywords have the most hits in `text`."""
    scores = {stage: 0 for stage in _STAGE_KEYWORDS}
    for stage, keywords in _STAGE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                # weight longer / more specific keywords more heavily
                scores[stage] += len(kw.split())
    best = max(scores, key=lambda s: scores[s])
    return best if scores[best] > 0 else "processing"


def _assign_tools(
    tools: list[dict],
    stages: dict[str, WorkflowStage],
) -> None:
    """
    Place each tool into the stage suggested by its category field.
    If the category is missing or unrecognised, fall back to keyword matching
    against the tool's purpose.
    """
    for tool in tools:
        name     = tool.get("name", "Unknown tool")
        category = tool.get("category", "")
        purpose  = tool.get("purpose", "")

        stage_key = _CATEGORY_TO_STAGE.get(category)
        if not stage_key:
            stage_key = _best_stage(purpose.lower())

        stages[stage_key].tools.append(name)


def _inject_llm_step(
    llm_rec: dict,
    stages: dict[str, WorkflowStage],
) -> None:
    """
    If the LLM recommendation says an LLM is used, inject a descriptive step
    into the processing stage (and record the model as a tool).
    """
    if not llm_rec.get("use_llm"):
        return

    model    = llm_rec.get("model_suggestion", "LLM")
    use_case = llm_rec.get("use_case", "")
    hint     = llm_rec.get("prompt_hint", "")

    step = f"Call {model}: {use_case}" if use_case else f"Call {model}"
    if hint:
        step += f' (prompt hint: "{hint}")'

    stages["processing"].steps.append(step)
    if model and model.lower() != "none" and model not in stages["processing"].tools:
        stages["processing"].tools.append(model)


def _fill_empty_stages(stages: dict[str, WorkflowStage]) -> None:
    """Add a placeholder step for any stage that ended up with no steps."""
    placeholders = {
        "trigger":          "Define the event or schedule that starts the automation",
        "input_collection": "Collect or fetch the required input data",
        "processing":       "Process and transform the collected data",
        "decision":         "Evaluate results and route accordingly",
        "output":           "Deliver the result to the target destination",
    }
    for key, stage in stages.items():
        if not stage.steps:
            stage.steps.append(placeholders[key])


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_response = {
        "decision": {
            "verdict": "Automate It",
            "confidence": "High",
            "reason": "Repetitive data entry with clear rules — strong automation candidate.",
        },
        "tools": [
            {"name": "Zapier",    "purpose": "Trigger on new form submission", "category": "Trigger"},
            {"name": "Airtable",  "purpose": "Store extracted data",           "category": "Storage"},
            {"name": "Slack",     "purpose": "Send approval notification",     "category": "Notification"},
            {"name": "GPT-4o mini","purpose": "Classify and summarise input",  "category": "AI"},
        ],
        "llm_recommendation": {
            "use_llm": True,
            "model_suggestion": "GPT-4o mini",
            "use_case": "Classify customer intent and summarise the request",
            "prompt_hint": "Classify the following request as Billing, Support, or Sales: {text}",
        },
        "workflow": [
            "Step 1: New form submission triggers the workflow via webhook",
            "Step 2: Fetch customer record from CRM",
            "Step 3: GPT-4o mini classifies and summarises the request",
            "Step 4: If score > 0.8, auto-approve; otherwise route for human review",
            "Step 5: Send Slack notification to the relevant team",
            "Step 6: Write result to Airtable",
        ],
    }

    wf = build_workflow(sample_response)
    print(wf.summary())
    print()
    import json
    print(json.dumps(wf.as_dict(), indent=2))
