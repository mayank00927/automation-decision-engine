"""
cost_calculator.py — Estimates monthly automation costs and break-even vs. human labour.
"""

from dataclasses import dataclass, field
from typing import Literal

# ── LLM pricing (per 1M tokens, USD) ─────────────────────────────────────────
# Input / output prices as of early 2026 (adjust as models reprice)
LLM_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":          {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":     {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":     {"input": 10.00, "output": 30.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":{"input": 3.00, "output": 15.00},
    "claude-haiku-4-5":{"input": 0.80,  "output": 4.00},
    "gemini-1.5-pro":  {"input": 3.50,  "output": 10.50},
    "gemini-1.5-flash":{"input": 0.075, "output": 0.30},
}

# ── Common tool subscription costs (USD/month) ────────────────────────────────
TOOL_COSTS: dict[str, float] = {
    "zapier":        29.0,
    "make":          9.0,
    "n8n":           20.0,   # cloud; self-hosted is free
    "airtable":      20.0,
    "notion":        10.0,
    "slack":         7.25,
    "hubspot":       50.0,
    "salesforce":    75.0,
    "github actions":0.0,    # free tier generous
    "aws lambda":    0.0,    # pay-per-use; captured in infra
    "google sheets": 0.0,    # free
    "sendgrid":      19.95,
    "twilio":        15.0,
    "jira":          8.15,
    "linear":        8.0,
    "pinecone":      70.0,
    "supabase":      25.0,
    "postgres":      0.0,    # self-hosted
    "redis":         0.0,    # self-hosted
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class LLMUsage:
    model: str = "gpt-4o-mini"
    daily_runs: int = 100
    avg_input_tokens: int = 500
    avg_output_tokens: int = 300

    def monthly_cost(self) -> float:
        key = self.model.lower()
        pricing = LLM_PRICING.get(key)
        if not pricing:
            # fall back to gpt-4o-mini pricing for unknown models
            pricing = LLM_PRICING["gpt-4o-mini"]

        monthly_runs = self.daily_runs * 30
        input_cost  = (self.avg_input_tokens  * monthly_runs / 1_000_000) * pricing["input"]
        output_cost = (self.avg_output_tokens * monthly_runs / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 4)


@dataclass
class InfraCost:
    """Cloud / self-hosted infrastructure monthly cost (USD)."""
    server_usd: float = 0.0      # e.g. VPS, EC2
    storage_usd: float = 0.0     # e.g. S3, GCS
    networking_usd: float = 0.0  # e.g. bandwidth, CDN

    def monthly_cost(self) -> float:
        return round(self.server_usd + self.storage_usd + self.networking_usd, 2)


@dataclass
class CostBreakdown:
    llm_monthly: float
    tools_monthly: float
    infra_monthly: float
    human_monthly_salary: float

    # derived
    total_automation_monthly: float = field(init=False)
    savings_monthly: float = field(init=False)
    break_even_months: float | None = field(init=False)

    # one-time setup cost fed externally
    setup_cost: float = 0.0

    def __post_init__(self) -> None:
        self.total_automation_monthly = round(
            self.llm_monthly + self.tools_monthly + self.infra_monthly, 2
        )
        self.savings_monthly = round(
            self.human_monthly_salary - self.total_automation_monthly, 2
        )
        if self.savings_monthly > 0:
            self.break_even_months = round(self.setup_cost / self.savings_monthly, 1)
        else:
            self.break_even_months = None  # never breaks even

    def roi_label(self) -> Literal["Excellent", "Good", "Marginal", "Poor"]:
        ratio = self.savings_monthly / max(self.total_automation_monthly, 0.01)
        if ratio >= 5:
            return "Excellent"
        if ratio >= 2:
            return "Good"
        if ratio >= 0.5:
            return "Marginal"
        return "Poor"

    def as_dict(self) -> dict:
        return {
            "llm_monthly_usd":          self.llm_monthly,
            "tools_monthly_usd":        self.tools_monthly,
            "infra_monthly_usd":        self.infra_monthly,
            "total_automation_monthly_usd": self.total_automation_monthly,
            "human_monthly_salary_usd": self.human_monthly_salary,
            "monthly_savings_usd":      self.savings_monthly,
            "setup_cost_usd":           self.setup_cost,
            "break_even_months":        self.break_even_months,
            "roi_label":                self.roi_label(),
        }


# ── Calculator ────────────────────────────────────────────────────────────────

class CostCalculator:
    """
    Orchestrates cost estimation for an automation project.

    Usage
    -----
    calc = CostCalculator(
        llm_usage=LLMUsage(model="gpt-4o-mini", daily_runs=200),
        tool_names=["zapier", "airtable"],
        infra=InfraCost(server_usd=10),
        human_monthly_salary=5000,
        setup_hours=8,
        dev_hourly_rate=75,
    )
    result = calc.calculate()
    print(result.as_dict())
    """

    def __init__(
        self,
        llm_usage: LLMUsage | None = None,
        tool_names: list[str] | None = None,
        extra_tool_costs: list[float] | None = None,
        infra: InfraCost | None = None,
        human_monthly_salary: float = 5000.0,
        setup_hours: float = 8.0,
        dev_hourly_rate: float = 75.0,
    ) -> None:
        self.llm_usage             = llm_usage or LLMUsage()
        self.tool_names            = [t.lower() for t in (tool_names or [])]
        self.extra_tool_costs      = extra_tool_costs or []
        self.infra                 = infra or InfraCost()
        self.human_monthly_salary  = human_monthly_salary
        self.setup_cost            = round(setup_hours * dev_hourly_rate, 2)

    # ── Public ─────────────────────────────────────────────────────────────────

    def calculate(self) -> CostBreakdown:
        return CostBreakdown(
            llm_monthly=self.llm_usage.monthly_cost(),
            tools_monthly=self._tools_monthly(),
            infra_monthly=self.infra.monthly_cost(),
            human_monthly_salary=self.human_monthly_salary,
            setup_cost=self.setup_cost,
        )

    def summary(self) -> str:
        b = self.calculate()
        d = b.as_dict()
        be = (
            f"  Break-even        : {d['break_even_months']} months"
            if d["break_even_months"] is not None
            else "  Break-even        : Never (costs exceed savings)"
        )
        lines = [
            "-- Automation Cost Summary ----------------------",
            f"  LLM cost          : ${d['llm_monthly_usd']:.2f}/mo",
            f"  Tool subscriptions: ${d['tools_monthly_usd']:.2f}/mo",
            f"  Infrastructure    : ${d['infra_monthly_usd']:.2f}/mo",
            f"  --------------------------------------------- ",
            f"  Total automation  : ${d['total_automation_monthly_usd']:.2f}/mo",
            f"  Human salary      : ${d['human_monthly_salary_usd']:.2f}/mo",
            f"  Monthly savings   : ${d['monthly_savings_usd']:.2f}",
            f"  Setup cost        : ${d['setup_cost_usd']:.2f}",
            be,
            f"  ROI label         : {d['roi_label']}",
            "-------------------------------------------------",
        ]
        return "\n".join(lines)

    # ── Private ────────────────────────────────────────────────────────────────

    def _tools_monthly(self) -> float:
        known   = sum(TOOL_COSTS.get(name, 0.0) for name in self.tool_names)
        unknown = sum(self.extra_tool_costs)
        return round(known + unknown, 2)


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    calc = CostCalculator(
        llm_usage=LLMUsage(model="gpt-4o-mini", daily_runs=150, avg_input_tokens=600, avg_output_tokens=400),
        tool_names=["zapier", "airtable", "slack"],
        infra=InfraCost(server_usd=10, storage_usd=2),
        human_monthly_salary=5000,
        setup_hours=8,
        dev_hourly_rate=75,
    )
    print(calc.summary())
