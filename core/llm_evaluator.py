# core/llm_evaluator.py
"""
Batch LLM evaluation of all human ISO 24027 answers.
Called once when the user clicks "Generate Report".

Each question is sent to Claude with full context:
  - dataset info (facet, outcome, rows)
  - question id, category, full text
  - user's Yes/No answer
  - user's free-text comment

Claude returns per question:
  {
    "is_risky":   bool,
    "risk_level": "Risky" | "Not Risky",
    "reason":     str,
    "confidence": float  0.0–1.0
  }

Auto questions (Q35/Q36/Q37) bypass the LLM entirely —
they are answered deterministically from DBDM metrics.
"""

import json
import urllib.request
import urllib.error
import os


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI bias auditor specialising in ISO/IEC 24027 \
(Bias in AI systems and AI aided decision making) and the DBDM (Data Bias Detection \
and Mitigation) framework.

Your task is to evaluate a practitioner's response to a single ISO 24027 data bias \
assessment question. You must judge whether the response demonstrates genuine, \
credible bias-awareness practice — or whether it indicates a risk of data bias \
being unaddressed.

Rules:
- A "No" answer to a best-practice question (e.g. "Do you use X?") is almost always Risky.
- A "Yes" answer with an empty or vague explanation (e.g. "yes we do it") is Risky \
  because it is an unsubstantiated claim.
- A "Yes" answer with a specific, named method, tool, or concrete process is Not Risky.
- For two-part questions (those containing "and how do you address..."), \
  BOTH parts must be addressed to be Not Risky.
- "Not answered" is always Risky.
- Be strict but fair. Domain-appropriate synonyms and paraphrases count as valid.

You must respond with ONLY valid JSON — no preamble, no markdown, no explanation outside \
the JSON object. The JSON must have exactly these four keys:
  "is_risky"   : boolean
  "risk_level" : "Risky" or "Not Risky"
  "reason"     : a single sentence (max 30 words) explaining your judgment
  "confidence" : a float between 0.0 and 1.0 reflecting how certain you are
"""


def _build_user_message(
    q: dict,
    answer: str,
    comment: str,
    dataset_info: dict,
) -> str:
    facet   = dataset_info.get("facet", "unknown")
    outcome = dataset_info.get("outcome", "unknown")
    rows    = dataset_info.get("rows", "unknown")
    subgroup = dataset_info.get("subgroup", "none")

    comment_text = comment.strip() if comment and comment.strip() else "(no explanation provided)"

    return f"""Dataset context:
- Sensitive attribute (facet): {facet}
- Outcome column: {outcome}
- Subgroup column: {subgroup}
- Number of rows: {rows}

Question ID: {q['id']}
Category: {q['category']}
Question type: {"describe — requires a named method" if q.get('q_type') == 'describe' else "confirm"}
Full question text: "{q['text']}"

Practitioner's answer: {answer}
Practitioner's explanation: {comment_text}

Evaluate this response and return JSON only."""


# ── Single question evaluation ────────────────────────────────────────────────

def _evaluate_one(q: dict, answer: str, comment: str, dataset_info: dict, api_key: str) -> dict:
    """
    Call Claude API for a single question.
    Returns the parsed JSON dict or a fallback error dict.
    """
    payload = json.dumps({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 256,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": _build_user_message(q, answer, comment, dataset_info),
            }
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw_text = body["content"][0]["text"].strip()

            # Strip accidental markdown fences
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            raw_text = raw_text.strip()

            parsed = json.loads(raw_text)

            # Normalise and validate
            return {
                "is_risky":   bool(parsed.get("is_risky", True)),
                "risk_level": str(parsed.get("risk_level", "Risky")),
                "reason":     str(parsed.get("reason", "No reason provided.")),
                "confidence": float(max(0.0, min(1.0, parsed.get("confidence", 0.5)))),
            }

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else str(e)
        return _error_result(f"API HTTP error {e.code}: {error_body[:120]}")
    except json.JSONDecodeError as e:
        return _error_result(f"Could not parse LLM response as JSON: {e}")
    except Exception as e:
        return _error_result(f"Unexpected error: {str(e)[:120]}")


def _error_result(reason: str) -> dict:
    return {
        "is_risky":   True,
        "risk_level": "Risky",
        "reason":     f"[Evaluation error] {reason}",
        "confidence": 0.0,
    }


# ── Batch evaluation (all human questions) ────────────────────────────────────

def evaluate_batch(
    human_questions: list,
    human_answers: dict,
    human_comments: dict,
    dataset_info: dict,
    api_key: str,
    progress_callback=None,
) -> dict:
    """
    Evaluate all human questions via Claude API.

    Args:
        human_questions  : list of question dicts (mode == "human")
        human_answers    : {qid: "Yes" / "No" / "Not answered"}
        human_comments   : {qid: str}
        dataset_info     : {"facet": str, "outcome": str, "rows": int, "subgroup": str}
        api_key          : Anthropic API key string
        progress_callback: optional callable(current, total, qid) for UI progress

    Returns:
        dict {qid: {is_risky, risk_level, reason, confidence}}
    """
    results = {}
    total = len(human_questions)

    for i, q in enumerate(human_questions):
        qid     = q["id"]
        answer  = human_answers.get(qid, "Not answered")
        comment = human_comments.get(qid, "")

        if progress_callback:
            progress_callback(i, total, qid)

        results[qid] = _evaluate_one(q, answer, comment, dataset_info, api_key)

    if progress_callback:
        progress_callback(total, total, "done")

    return results
