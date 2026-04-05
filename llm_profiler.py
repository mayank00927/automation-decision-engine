"""
llm_profiler.py — Identifies the LLM used in a workflow and extracts its
active capability features (text generation, reasoning, classification,
automation) from the workflow's use-case description, steps, and tool purposes.
"""

# ── Provider registry ──────────────────────────────────────────────────────────
# Ordered: more-specific prefixes first.

_PROVIDER_MAP: list[tuple[str, str]] = [
    ("claude-",       "Anthropic"),
    ("gpt-",          "OpenAI"),
    ("o1",            "OpenAI"),
    ("o3",            "OpenAI"),
    ("gemini-",       "Google"),
    ("mistral",       "Mistral AI"),
    ("llama",         "Meta"),
    ("command",       "Cohere"),
    ("titan",         "Amazon"),
    ("palm",          "Google"),
]

# ── Capability keywords ────────────────────────────────────────────────────────
# Lowercase substrings; partial matches are intentional (e.g. "classif" matches
# "classify", "classification", "classifier").

_FEATURE_KEYWORDS: dict[str, list[str]] = {
    "text_generation": [
        "generat", "write", "draft", "summariz", "summaris",
        "compose", "create content", "produce text", "describe",
        "explain", "translat", "rewrite", "paraphras",
        "respond", "reply", "answer", "narrat", "report",
    ],
    "reasoning": [
        "reason", "decid", "evaluat", "assess", "scor",
        "rout", "analyz", "analys", "compar", "judg",
        "infer", "deduc", "prioritiz", "prioritis",
        "recommend", "suggest", "strateg", "plan",
    ],
    "classification": [
        "classif", "categori", "label", "tag ",
        "sort ", "filter", "identif", "detect",
        "match", "group", "bin ", " type",
        "sentiment", "intent", "triage",
    ],
    "automation": [
        "automat", "trigger", "schedul", "orchestrat",
        "chain", "pipeline", "workflow", "execut",
        "batch", "loop", "monitor", "watch", "poll",
        "dispatch", "handl", "process",
    ],
}


# ── Public API ─────────────────────────────────────────────────────────────────

def profile_llm(wf: dict) -> dict:
    """
    Analyse a single workflow dict and return a structured LLM profile.

    Parameters
    ----------
    wf : dict
        One workflow entry as returned by the LLM (keys: llm_model,
        llm_use_case, workflow_steps, tools, estimated_daily_runs, …).

    Returns
    -------
    dict with keys:
        model               – e.g. "gpt-4o-mini"
        provider            – e.g. "OpenAI"
        use_case            – free-text description of the LLM's role
        features            – {text_generation, reasoning, classification,
                               automation} → bool
        feature_evidence    – {feature_name → matched keyword or excerpt}
        token_profile       – {estimated_daily_runs, avg_input_tokens,
                               avg_output_tokens}
    """
    model = wf.get("llm_model", "none")

    if model.lower() == "none":
        return _no_llm_profile()

    use_case     = wf.get("llm_use_case", "")
    steps        = " ".join(wf.get("workflow_steps", []))
    ai_purposes  = " ".join(
        t.get("purpose", "")
        for t in wf.get("tools", [])
        if t.get("category") == "AI"
    )
    corpus = f"{use_case} {steps} {ai_purposes}".lower()

    features, evidence = _extract_features(corpus)

    return {
        "model":            model,
        "provider":         _detect_provider(model),
        "use_case":         use_case,
        "features":         features,
        "feature_evidence": evidence,
        "token_profile": {
            "estimated_daily_runs": wf.get("estimated_daily_runs", 0),
            "avg_input_tokens":     wf.get("avg_input_tokens", 0),
            "avg_output_tokens":    wf.get("avg_output_tokens", 0),
        },
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _no_llm_profile() -> dict:
    return {
        "model":    "none",
        "provider": "none",
        "use_case": "",
        "features": {
            "text_generation": False,
            "reasoning":       False,
            "classification":  False,
            "automation":      False,
        },
        "feature_evidence": {},
        "token_profile": {
            "estimated_daily_runs": 0,
            "avg_input_tokens":     0,
            "avg_output_tokens":    0,
        },
    }


def _detect_provider(model: str) -> str:
    """Return the provider name for a known model, or 'Unknown' otherwise."""
    lower = model.lower()
    for prefix, provider in _PROVIDER_MAP:
        if lower.startswith(prefix):
            return provider
    return "Unknown"


def _extract_features(corpus: str) -> tuple[dict[str, bool], dict[str, str]]:
    """
    Scan corpus for feature keywords.

    Returns
    -------
    features : {feature_name → bool}
    evidence : {feature_name → first matched keyword}
    """
    features: dict[str, bool] = {}
    evidence: dict[str, str]  = {}

    for feature, keywords in _FEATURE_KEYWORDS.items():
        hit = next((kw for kw in keywords if kw in corpus), None)
        features[feature] = hit is not None
        if hit is not None:
            evidence[feature] = _excerpt(corpus, hit)

    return features, evidence


def _excerpt(corpus: str, keyword: str, context: int = 40) -> str:
    """Return a short excerpt from corpus centred on the matched keyword."""
    idx = corpus.find(keyword)
    if idx == -1:
        return keyword.strip()
    start = max(0, idx - context)
    end   = min(len(corpus), idx + len(keyword) + context)
    snippet = corpus[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(corpus):
        snippet = snippet + "…"
    return snippet
