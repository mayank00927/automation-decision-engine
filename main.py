import asyncio
import uuid
import json
import sqlite3
import logging
from pathlib import Path
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from llm_service import LLMService
from cost_calculator import CostCalculator, LLMUsage, InfraCost
from llm_profiler import profile_llm
from web_search_service import search_tools, search_pricing

logger = logging.getLogger(__name__)

app = FastAPI(title="Automation Decision Engine")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

llm = LLMService()

# ── Persistent SQLite store ────────────────────────────────────────────────────

_DB_PATH = Path("dashboards.db")


def _init_db() -> None:
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS dashboards (
                idea_id    TEXT PRIMARY KEY,
                idea       TEXT NOT NULL,
                data       TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )"""
        )


_init_db()


def _store_dashboard(idea_id: str, dashboard: dict) -> None:
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO dashboards (idea_id, idea, data) VALUES (?,?,?)",
            (idea_id, dashboard.get("idea", ""), json.dumps(dashboard)),
        )


def _load_dashboard(idea_id: str) -> dict | None:
    with sqlite3.connect(_DB_PATH) as conn:
        row = conn.execute(
            "SELECT data FROM dashboards WHERE idea_id = ?", (idea_id,)
        ).fetchone()
    return json.loads(row[0]) if row else None


SYSTEM_PROMPT = """You are an automation strategy expert. Analyze the user's automation idea and return a JSON object with exactly these fields:

{
  "decision": {
    "verdict": "Automate It | Needs Review | Skip It",
    "confidence": "High | Medium | Low",
    "reason": "One or two sentences explaining the verdict"
  },
  "tools": [
    {"name": "Tool Name", "purpose": "What it does in this workflow", "category": "Trigger | Action | AI | Storage | Notification"}
  ],
  "llm_recommendation": {
    "use_llm": true,
    "model_suggestion": "e.g. GPT-4o mini, Claude Haiku, none",
    "use_case": "What the LLM would do in this automation",
    "prompt_hint": "A short example of the kind of prompt you would use"
  },
  "workflow": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "cost_roi": {
    "estimated_setup_hours": "e.g. 4-8 hours",
    "estimated_time_saved_weekly": "e.g. 6 hours/week",
    "estimated_monthly_tool_cost": "e.g. $20-$50/month",
    "break_even": "e.g. 2-3 weeks",
    "roi_score": "Excellent | Good | Marginal | Poor",
    "notes": "Any important cost or ROI caveat"
  }
}

Return ONLY valid JSON. No markdown, no extra text."""


class AnalyzeRequest(BaseModel):
    idea: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@app.post("/analyze")
async def analyze(body: AnalyzeRequest):
    if not body.idea.strip():
        raise HTTPException(status_code=422, detail="idea must not be empty")

    outcome = llm.analyze(f"Automation idea: {body.idea}", SYSTEM_PROMPT)
    if outcome.success:
        return JSONResponse(content=outcome.data)

    raise HTTPException(status_code=500, detail=outcome.error)


# ── Dashboard: 5-workflow system prompt ───────────────────────────────────────

# One variation per parallel LLM call — keeps each response small and focused.
_WORKFLOW_VARIATIONS: list[tuple[str, str]] = [
    ("Full AI Pipeline",     "Use heavy LLM involvement for intelligence and decision-making."),
    ("No-Code / Low-Code",   "Use no-code tools like Zapier, Make, or n8n — minimal or no custom code."),
    ("Custom Code",          "Use Python scripts, serverless functions, or custom APIs for full control."),
    ("AI + No-Code Hybrid",  "Combine a no-code platform with a lightweight LLM call for smart routing."),
    ("Minimal / Rule-Based", "Use simple conditional rules and filters — no LLM, minimal tooling."),
]

_SINGLE_WORKFLOW_PROMPT = """You are an automation architect. Given an automation idea and a specific approach constraint, generate exactly ONE workflow that fits that approach.

Return ONLY a valid JSON object (no extra text):
{
  "title": "Short descriptive name matching the approach type",
  "approach": "One-line summary, e.g. 'Full AI pipeline with GPT-4o'",
  "tools": [
    {
      "name": "ToolName",
      "purpose": "What this tool does in the workflow",
      "category": "Trigger|Action|AI|Storage|Notification"
    }
  ],
  "llm_model": "gpt-4o-mini|gpt-4o|claude-haiku-4-5|claude-sonnet-4-6|none",
  "llm_use_case": "What the LLM does, or empty string if no LLM",
  "estimated_daily_runs": 100,
  "avg_input_tokens": 500,
  "avg_output_tokens": 300,
  "workflow_steps": ["Step 1: ...", "Step 2: ..."],
  "scalability_score": 7,
  "rationale": "Why someone would choose this option"
}

Rules:
- If llm_model is "none", set estimated_daily_runs/avg_input_tokens/avg_output_tokens to 0.
- scalability_score: integer 1-10 (10 = trivially scales to 10x volume, 1 = hard ceiling).
- Stay true to the approach constraint given in the user message."""


# ── Dashboard Pydantic models ──────────────────────────────────────────────────

class IdeaRequest(BaseModel):
    idea:        str  = Field(..., min_length=10, description="Automation idea or problem statement")
    use_realtime: bool = Field(False, description="Fetch latest tools and pricing via web search before generating workflows")


class IdeaResponse(BaseModel):
    idea_id: str
    message: str


# ── Web-context helpers ────────────────────────────────────────────────────────

def _build_web_context_block(tool_results: list, pricing_results: list) -> str:
    """Format search results into a prompt block for LLM injection."""
    lines = ["--- REAL-TIME WEB CONTEXT (use this to improve your answer) ---"]

    if tool_results:
        lines.append("\n[Latest Tools & Integrations]")
        for r in tool_results:
            lines.append(f"• {r['tool_name']} — {r['latest_price']} — {r['source']}")

    if pricing_results:
        lines.append("\n[Latest Pricing]")
        for r in pricing_results:
            lines.append(f"• {r['tool_name']}: {r['latest_price']} — {r['source']}")

    lines.append("--- END OF WEB CONTEXT ---")
    return "\n".join(lines)


# ── Dashboard helpers ──────────────────────────────────────────────────────────

def _compute_cost(wf: dict) -> dict:
    tool_names = [t.get("name", "").lower() for t in wf.get("tools", [])]
    llm_model  = wf.get("llm_model", "none")
    use_llm    = llm_model.lower() != "none"

    llm_usage = LLMUsage(
        model=llm_model,
        daily_runs=wf.get("estimated_daily_runs", 0),
        avg_input_tokens=wf.get("avg_input_tokens", 0),
        avg_output_tokens=wf.get("avg_output_tokens", 0),
    ) if use_llm else LLMUsage(model="gpt-4o-mini", daily_runs=0)

    calc = CostCalculator(
        llm_usage=llm_usage,
        tool_names=tool_names,
        infra=InfraCost(),
        human_monthly_salary=5000.0,
        setup_hours=8.0,
        dev_hourly_rate=75.0,
    )
    return calc.calculate().as_dict()


def _minmax(values: list[float], invert: bool = False) -> list[float]:
    """Min-max normalise a list to [0, 1]. Invert so lower raw = higher score."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    normed = [(v - lo) / (hi - lo) for v in values]
    return [1.0 - n for n in normed] if invert else normed


def _score_workflows(
    costed: list[tuple[dict, dict]],
) -> list[tuple[dict, dict, float, dict]]:
    """
    Score each workflow on four dimensions and return
    (wf, cost, composite_score, score_breakdown) tuples sorted best → worst.

    Weights
    -------
    ROI          40 %  – savings-to-cost ratio (higher is better)
    Cost         30 %  – total monthly automation spend (lower is better)
    Complexity   20 %  – steps + tool count (fewer is better)
    Scalability  10 %  – LLM-provided score 1-10 (higher is better)
    """
    roi_raw, cost_raw, complexity_raw, scale_raw = [], [], [], []

    for wf, cost in costed:
        roi_raw.append(
            cost["monthly_savings_usd"] / max(cost["total_automation_monthly_usd"], 0.01)
        )
        cost_raw.append(cost["total_automation_monthly_usd"])
        complexity_raw.append(
            len(wf.get("workflow_steps", [])) + len(wf.get("tools", []))
        )
        scale_raw.append(float(wf.get("scalability_score", 5)))

    roi_n   = _minmax(roi_raw,        invert=False)
    cost_n  = _minmax(cost_raw,       invert=True)   # lower spend → higher score
    comp_n  = _minmax(complexity_raw, invert=True)   # fewer steps → higher score
    scale_n = _minmax(scale_raw,      invert=False)

    scored = []
    for i, (wf, cost) in enumerate(costed):
        breakdown = {
            "roi_score":         round(roi_n[i],   4),
            "cost_score":        round(cost_n[i],  4),
            "complexity_score":  round(comp_n[i],  4),
            "scalability_score": round(scale_n[i], 4),
        }
        composite = round(
            0.40 * roi_n[i]
            + 0.30 * cost_n[i]
            + 0.20 * comp_n[i]
            + 0.10 * scale_n[i],
            4,
        )
        scored.append((wf, cost, composite, breakdown))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def _build_dashboard(idea_id: str, idea: str, llm_data: dict) -> dict:
    raw = [wf for wf in llm_data.get("workflows", []) if isinstance(wf, dict)][:5]
    if not raw:
        raise ValueError("LLM returned no workflows. Please try again with a more specific idea.")

    costed = [(wf, _compute_cost(wf)) for wf in raw]
    scored = _score_workflows(costed)

    workflows = []
    for rank, (wf, cost, composite, score_breakdown) in enumerate(scored, start=1):
        workflows.append({
            "ranking":         rank,
            "composite_score": composite,
            "score_breakdown": score_breakdown,
            "title":    wf.get("title", f"Option {rank}"),
            "approach": wf.get("approach", ""),
            "tools":    wf.get("tools", []),
            "llm_details":    profile_llm(wf),
            "workflow_steps": wf.get("workflow_steps", []),
            "cost_breakdown": cost,
            "rationale":      wf.get("rationale", ""),
        })

    return {"idea_id": idea_id, "idea": idea, "workflows": workflows}


# ── Dashboard endpoints ────────────────────────────────────────────────────────

@app.post("/ideas", response_model=IdeaResponse, status_code=201)
async def submit_idea(body: IdeaRequest):
    """
    Submit an automation idea.

    Fires one LLM call per workflow variation in parallel (5 concurrent calls,
    each capped at 1 500 tokens) instead of one large call.  This avoids JSON
    truncation and keeps each response well within token limits.

    When use_realtime=true the backend runs two web searches (latest tools,
    latest pricing) and injects the results into every prompt before
    generating the workflows.  When false, only static/trained knowledge is used.

    Results are stored in SQLite and retrievable via GET /dashboard/{idea_id}.
    """
    system_prompt = _SINGLE_WORKFLOW_PROMPT

    # ── Real-time enrichment ───────────────────────────────────────────────────
    if body.use_realtime:
        tool_results    = search_tools(body.idea)
        top_names       = [r["tool_name"] for r in tool_results[:3]]
        pricing_results = [search_pricing(name) for name in top_names]

        if tool_results or pricing_results:
            web_block     = _build_web_context_block(tool_results, pricing_results)
            system_prompt = _SINGLE_WORKFLOW_PROMPT + "\n\n" + web_block
            logger.info(
                "Real-time context injected: %d tool results, %d pricing results",
                len(tool_results), len(pricing_results),
            )
        else:
            logger.warning(
                "web_search_toggle=true but no results returned "
                "(Tavily key missing and DDG unavailable). Using static knowledge."
            )

    # ── Parallel batch: one call per variation ─────────────────────────────────
    loop = asyncio.get_running_loop()

    def _call(variation_name: str, variation_hint: str):
        user_msg = (
            f"Automation idea: {body.idea}\n"
            f"Approach constraint: {variation_name} — {variation_hint}"
        )
        return llm.analyze(user_msg, system_prompt, max_tokens=1500)

    results = await asyncio.gather(*[
        loop.run_in_executor(None, _call, name, hint)
        for name, hint in _WORKFLOW_VARIATIONS
    ])

    failed = [r for r in results if r.failed]
    if failed:
        raise HTTPException(status_code=502, detail=failed[0].error)

    llm_data = {"workflows": [r.data for r in results]}

    idea_id = str(uuid.uuid4())
    try:
        dashboard = _build_dashboard(idea_id, body.idea, llm_data)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    dashboard["realtime_data_used"] = body.use_realtime
    _store_dashboard(idea_id, dashboard)
    logger.info("Dashboard stored for idea_id=%s (realtime=%s)", idea_id, body.use_realtime)

    return IdeaResponse(
        idea_id=idea_id,
        message="Analysis complete. Retrieve results at GET /dashboard/{idea_id}.",
    )


@app.get("/dashboard/{idea_id}")
async def get_dashboard(idea_id: str):
    """
    Return 5 ranked workflows for a previously submitted idea.

    Each workflow entry includes:
    - ranking          (1 = highest composite score)
    - composite_score  (weighted 0-1)
    - score_breakdown  (roi 40 %, cost 30 %, complexity 20 %, scalability 10 %)
    - tools            (name, purpose, category)
    - cost_breakdown   (LLM + tool subscriptions + infra, roi_label, break_even_months)
    - llm_details      (model, use_case, token estimates)
    """
    dashboard = _load_dashboard(idea_id)
    if dashboard is None:
        raise HTTPException(
            status_code=404,
            detail=f"No dashboard found for idea_id '{idea_id}'. Submit the idea first via POST /ideas.",
        )
    return JSONResponse(content=dashboard)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
