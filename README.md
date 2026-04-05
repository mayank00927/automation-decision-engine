# Automation Decision Engine

A FastAPI web app that takes any automation idea and returns a ranked comparison of 5 workflow approaches — with tool recommendations, cost breakdowns, and ROI scoring.

## What it does

Paste an automation idea (e.g. *"When a new lead fills our Typeform, score them and send a personalised email"*) and the engine:

1. Generates **5 parallel workflow variations** via LLM — Full AI Pipeline, No-Code, Custom Code, AI+No-Code Hybrid, and Rule-Based
2. **Scores and ranks** each workflow on ROI (40%), cost (30%), complexity (20%), and scalability (10%)
3. Returns a **cost breakdown** — LLM token costs, tool subscriptions, infra, break-even months
4. Optionally enriches results with **real-time tool/pricing data** via web search (Tavily or DuckDuckGo)

## Tech stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI + Uvicorn |
| LLM | OpenAI GPT-4o mini (via `openai` SDK) |
| Frontend | Jinja2 templates + Tailwind CSS |
| Persistence | SQLite |
| Web Search | Tavily API (optional) / DuckDuckGo fallback |

## Project structure

```
├── main.py                 # FastAPI app, endpoints, dashboard logic
├── llm_service.py          # OpenAI wrapper — always returns structured JSON
├── cost_calculator.py      # LLM + tool + infra cost model
├── llm_profiler.py         # Token usage profiling per workflow
├── web_search_service.py   # Tavily / DuckDuckGo search for real-time pricing
├── workflow_builder.py     # Workflow assembly helpers
├── models.py               # Shared Pydantic models
├── templates/
│   ├── index.html          # Single-idea analysis page
│   └── dashboard.html      # 5-workflow comparison dashboard
├── requirements.txt
└── .env.example
```

## Setup

**1. Clone and install**

```bash
git clone https://github.com/mayank00927/automation-decision-engine.git
cd automation-decision-engine
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure environment**

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```
OPENAI_API_KEY=sk-...          # Required
TAVILY_API_KEY=tvly-...        # Optional — enables real-time web search
```

**3. Run**

```bash
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Single-idea analysis page |
| `POST` | `/analyze` | Quick analysis — returns decision, tools, workflow, ROI |
| `GET` | `/dashboard` | 5-workflow comparison dashboard |
| `POST` | `/ideas` | Submit idea → generates 5 ranked workflows, stores in SQLite |
| `GET` | `/dashboard/{idea_id}` | Retrieve stored dashboard by ID |

### POST /ideas — request body

```json
{
  "idea": "Auto-triage support emails and create Jira tickets",
  "use_realtime": false
}
```

Set `use_realtime: true` to fetch latest tool pricing from the web before generating workflows.

## Workflow scoring

Each of the 5 workflows is scored on a 0–1 composite:

- **ROI (40%)** — monthly savings ÷ automation cost
- **Cost (30%)** — total monthly spend (lower is better)
- **Complexity (20%)** — step count + tool count (fewer is better)
- **Scalability (10%)** — LLM-provided score 1–10

## License

MIT
