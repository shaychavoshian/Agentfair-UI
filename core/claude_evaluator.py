# core/claude_evaluator.py
"""
Claude-powered evaluation for AgentFair ISO 24027 questionnaire.

Two public functions:
  - generate_suggested_answer(question, dbdm_metrics, dataset_summary) -> str
  - evaluate_user_answer(question, user_answer, dbdm_metrics, dataset_summary) -> dict

All API calls go to the Anthropic /v1/messages endpoint.
The API key is injected by the Anthropic environment (no key needed in code).
"""

import json
import requests

_API_URL = "https://api.anthropic.com/v1/messages"
_MODEL   = "claude-sonnet-4-5-20250929"
_HEADERS = {
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
}

# ── Threshold display helper (mirrors questionnaire.py) ───────────────────────
def _fmt_metrics(dbdm_metrics: dict, threshold_display: dict) -> str:
    """Return a compact readable string of metric values and their thresholds."""
    lines = []
    for name, val in dbdm_metrics.items():
        thr = threshold_display.get(name, "±0.1")
        lines.append(f"  • {name} = {round(float(val), 4)}  (threshold: {thr})")
    return "\n".join(lines) if lines else "  (no metrics available)"


def _call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
    """Make a single call to the Anthropic Messages API and return the text."""
    payload = {
        "model": _MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    resp = requests.post(_API_URL, headers=_HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"].strip()


# ── System prompts ─────────────────────────────────────────────────────────────
_SUGGEST_SYSTEM = """You are an expert AI fairness auditor specialising in ISO/IEC 24027 
(Bias in AI systems and AI aided decision making) and the DBDM fairness metric toolkit.

Your task is to generate a concise, expert suggested answer to a single ISO 24027 
assessment question, given:
- The question text and its bias category
- The DBDM fairness metrics computed from the dataset
- A brief summary of the dataset

Your suggested answer should:
1. Be 3–5 sentences
2. Directly address what the question is asking
3. Reference specific DBDM metric values where relevant
4. Be written as if YOU are the auditor making a recommendation
5. Not use bullet points — write in clear prose

Do not include disclaimers or preamble. Just give the answer directly."""


_EVALUATE_SYSTEM = """You are an expert AI fairness auditor specialising in ISO/IEC 24027 
and the DBDM fairness metric toolkit. You are evaluating a practitioner's answer to an 
ISO 24027 bias assessment question.

You must return ONLY a valid JSON object with exactly these fields:
{
  "score": <integer 0-10>,
  "verdict": "<Risky|Not Risky>",
  "feedback": "<2-4 sentence evaluation of the user's answer>",
  "key_gap": "<one sentence describing the most important thing missing, or 'None' if complete>"
}

Scoring guide:
  0-3  : Answer is absent, vague, or incorrect — clear bias risk
  4-6  : Partial answer — some awareness but missing key methods or evidence
  7-8  : Good answer — practice is in place with reasonable justification
  9-10 : Excellent — specific method named, evidence or metric cited, thorough

Verdict rule:
  score >= 7  → "Not Risky"
  score < 7   → "Risky"

Base your evaluation on:
- The DBDM metric values provided (they give quantitative context)
- Whether the user describes a concrete, recognised method or practice
- Whether their answer is consistent with what the metrics show
- ISO 24027 best practices for this bias category

Return ONLY the JSON. No preamble, no markdown fences."""


# ── Public API ────────────────────────────────────────────────────────────────

def generate_suggested_answer(
    question: dict,
    dbdm_metrics: dict,
    threshold_display: dict,
    dataset_summary: dict,
) -> str:
    """
    Generate Claude's expert suggested answer for a given ISO 24027 question.

    Parameters
    ----------
    question         : question dict from QUESTIONS list
    dbdm_metrics     : flat {metric_name: float} dict from DBDM
    threshold_display: {metric_name: threshold_string} for display
    dataset_summary  : {"rows": int, "facet": str, "outcome": str, "subgroup": str}

    Returns
    -------
    Suggested answer string, or an error message string.
    """
    metric_block = _fmt_metrics(dbdm_metrics, threshold_display)
    ds = dataset_summary

    user_prompt = f"""ISO 24027 Bias Category: {question['category']}

Question ({question['id']}):
{question['text']}

Dataset context:
  Rows: {ds.get('rows', 'unknown')}
  Sensitive attribute (facet): {ds.get('facet', 'unknown')}
  Outcome column: {ds.get('outcome', 'unknown')}
  Subgroup column: {ds.get('subgroup', 'none')}

DBDM Fairness Metrics computed from this dataset:
{metric_block}

Please provide your expert suggested answer to this question."""

    try:
        return _call_claude(_SUGGEST_SYSTEM, user_prompt, max_tokens=400)
    except Exception as e:
        return f"[Could not generate suggestion: {e}]"


def evaluate_user_answer(
    question: dict,
    user_answer: str,
    dbdm_metrics: dict,
    threshold_display: dict,
    dataset_summary: dict,
) -> dict:
    """
    Evaluate a practitioner's free-text answer using Claude.

    Returns
    -------
    dict with keys: score (int), verdict (str), feedback (str), key_gap (str)
    On API failure, returns a safe fallback dict.
    """
    metric_block = _fmt_metrics(dbdm_metrics, threshold_display)
    ds = dataset_summary

    user_prompt = f"""ISO 24027 Bias Category: {question['category']}

Question ({question['id']}):
{question['text']}

Dataset context:
  Rows: {ds.get('rows', 'unknown')}
  Sensitive attribute (facet): {ds.get('facet', 'unknown')}
  Outcome column: {ds.get('outcome', 'unknown')}
  Subgroup column: {ds.get('subgroup', 'none')}

DBDM Fairness Metrics:
{metric_block}

Practitioner's answer:
\"\"\"{user_answer.strip() if user_answer else '(no answer provided)'}\"\"\"

Evaluate this answer and return the JSON object as specified."""

    try:
        raw = _call_claude(_EVALUATE_SYSTEM, user_prompt, max_tokens=400)
        # Strip accidental markdown fences if present
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(raw)
        # Enforce types
        result["score"]   = max(0, min(10, int(result.get("score", 0))))
        result["verdict"] = "Not Risky" if result["score"] >= 7 else "Risky"
        result["feedback"] = str(result.get("feedback", ""))
        result["key_gap"]  = str(result.get("key_gap", ""))
        return result
    except Exception as e:
        return {
            "score":   0,
            "verdict": "Risky",
            "feedback": f"Evaluation could not be completed: {e}",
            "key_gap":  "Unable to assess.",
        }
