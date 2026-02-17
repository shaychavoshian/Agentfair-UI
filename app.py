# app.py  –  AgentFair Interactive Bias Detection UI
# Run:  streamlit run app.py

import streamlit as st
import pandas as pd
import tempfile
import os
import json
import sys

sys.path.insert(0, os.path.dirname(__file__))

from core.questionnaire import (
    QUESTIONS, DBDM_THRESHOLDS, THRESHOLD_DISPLAY,
    CATEGORY_METRICS, evaluate_all, auto_answer, _metric_is_risky,
)
from core.dbdm_runner import run_dbdm
from core.claude_evaluator import generate_suggested_answer, evaluate_user_answer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgentFair – ISO 24027 Bias Detection",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.header-strip {
    background: linear-gradient(90deg,#1a237e,#283593);
    color:white; padding:16px 24px; border-radius:8px; margin-bottom:20px;
}
.risky-box {
    background:#fff3cd; border-left:4px solid #e53935;
    padding:12px 16px; border-radius:6px; margin-bottom:8px;
}
.safe-box {
    background:#e8f5e9; border-left:4px solid #43a047;
    padding:12px 16px; border-radius:6px; margin-bottom:8px;
}
.auto-box {
    background:#e3f2fd; border-left:4px solid #1e88e5;
    padding:12px 16px; border-radius:6px; margin-bottom:8px;
}
.score-chip-risky {
    display:inline-block; background:#e53935; color:white;
    border-radius:12px; padding:2px 12px; font-weight:700; font-size:0.95rem;
}
.score-chip-safe {
    display:inline-block; background:#43a047; color:white;
    border-radius:12px; padding:2px 12px; font-weight:700; font-size:0.95rem;
}
.q-card {
    background:#f0f4ff; border:1px solid #c5cae9;
    border-radius:10px; padding:20px 24px; margin:12px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "step": 1,
        "df": None,
        "csv_path": None,
        "facet": None,
        "outcome": None,
        "subgroup": "",
        "label_value": 1.0,
        "dbdm_metrics": {},
        "dbdm_json_path": None,
        "human_answers": {},
        "all_results": [],
        "claude_evals_done": False,
        "dbdm_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def _dataset_summary():
    return {
        "rows":     len(st.session_state.df) if st.session_state.df is not None else "unknown",
        "facet":    st.session_state.facet   or "unknown",
        "outcome":  st.session_state.outcome or "unknown",
        "subgroup": st.session_state.subgroup or "none",
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ AgentFair")
    st.markdown("*ISO 24027 + DBDM + Claude AI*")
    st.divider()
    for num, label in [(1,"Upload Dataset"),(2,"DBDM Metrics"),
                        (3,"ISO Questionnaire"),(4,"Bias Report")]:
        cur  = st.session_state.step == num
        done = st.session_state.step > num
        icon = "✅" if done else ("▶️" if cur else "○")
        txt  = f"**Step {num}: {label}**" if cur else f"Step {num}: {label}"
        st.markdown(f"{icon} {txt}")

    if st.session_state.step >= 3:
        human_qs = [q for q in QUESTIONS if q["mode"] == "human"]
        answered = sum(1 for q in human_qs
                       if st.session_state.human_answers.get(q["id"], "").strip())
        st.divider()
        st.markdown("**Questionnaire progress**")
        st.progress(answered / max(len(human_qs), 1))
        st.caption(f"{answered} / {len(human_qs)} answered")

    if st.session_state.step > 1:
        st.divider()
        if st.button("🔄 Start Over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.markdown("""
    <div class='header-strip'>
        <h2 style='margin:0;color:white;'>⚖️ AgentFair — AI Bias Detection</h2>
        <p style='margin:4px 0 0;color:#b3c5f8;'>ISO 24027 + DBDM Toolkit + Claude AI Evaluation</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Step 1 — Upload your dataset")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        st.success(f"✅ Loaded **{len(df):,}** rows × **{len(df.columns)}** columns")
        st.dataframe(df.head(5), use_container_width=True)
        st.divider()

        cols = list(df.columns)
        c1, c2, c3 = st.columns(3)
        with c1:
            facet = st.selectbox("Sensitive attribute (facet)", cols,
                help="Protected/demographic attribute, e.g. 'vendor', 'gender'")
        with c2:
            outcome = st.selectbox("Outcome column",
                [c for c in cols if c != facet],
                help="The label/target column")
        with c3:
            subgroup = st.selectbox("Subgroup column (optional)",
                ["(none)"] + [c for c in cols if c not in [facet, outcome]])

        label_value = st.number_input("Positive label value", value=1.0, step=0.5,
            help="Value in the outcome column that counts as a 'positive'")

        if st.button("▶️ Continue →", type="primary"):
            st.session_state.facet       = facet
            st.session_state.outcome     = outcome
            st.session_state.subgroup    = "" if subgroup == "(none)" else subgroup
            st.session_state.label_value = label_value
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            df.to_csv(tmp.name, index=False)
            st.session_state.csv_path = tmp.name
            st.session_state.step = 2
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DBDM METRICS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    st.markdown("### ⚙️ Step 2 — DBDM Bias Metrics")

    if not st.session_state.dbdm_metrics and not st.session_state.dbdm_error:
        with st.spinner("Running DBDM — calculating 15 fairness metrics…"):
            try:
                metrics, json_path = run_dbdm(
                    file_path=st.session_state.csv_path,
                    facet=st.session_state.facet,
                    outcome=st.session_state.outcome,
                    subgroup_col=st.session_state.subgroup,
                    label_value=st.session_state.label_value,
                )
                st.session_state.dbdm_metrics   = metrics
                st.session_state.dbdm_json_path = json_path
            except Exception as e:
                st.session_state.dbdm_error = str(e)

    if st.session_state.dbdm_error:
        st.error(f"❌ DBDM failed: {st.session_state.dbdm_error}")
        if st.button("← Back"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

    metrics = st.session_state.dbdm_metrics
    st.success("✅ Metrics calculated successfully.")

    st.markdown("#### All DBDM Metrics")
    rows = []
    for name, val in metrics.items():
        risky = _metric_is_risky(name, val)
        thr   = THRESHOLD_DISPLAY.get(name, "±0.1")
        rows.append({
            "Metric": name,
            "Value": round(float(val), 4) if val is not None else "—",
            "Threshold": thr,
            "Status": "⚠️ RISKY" if risky else "✅ FAIR",
        })
    mdf = pd.DataFrame(rows)
    def color_status(row):
        if row["Status"].startswith("⚠️"):
            return ["background-color:#fff3cd"] * len(row)
        return ["background-color:#e8f5e9"] * len(row)
    st.dataframe(mdf.style.apply(color_status, axis=1),
                 use_container_width=True, hide_index=True)

    risky_count = sum(1 for r in rows if r["Status"].startswith("⚠️"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Metrics", len(rows))
    c2.metric("⚠️ Outside Threshold", risky_count)
    c3.metric("✅ Within Threshold", len(rows) - risky_count)

    st.divider()
    st.markdown("#### Auto-answered Questions (Q35, Q36, Q37)")
    for q in [q for q in QUESTIONS if q["mode"] == "auto"]:
        result = auto_answer(q, metrics)
        icon   = "⚠️" if result["is_risky"] else "✅"
        label  = "RISKY" if result["is_risky"] else "NOT RISKY"
        css    = "risky-box" if result["is_risky"] else "auto-box"
        mparts = []
        for m, v in result["values"].items():
            thr     = THRESHOLD_DISPLAY.get(m, "±0.1")
            outside = m in result["risky_metrics"]
            flag    = "⚠️" if outside else "✅"
            mparts.append(f"{flag} &nbsp;<b>{m}</b> = {v} &nbsp;(threshold {thr})")
        metric_html = "<br>".join(mparts)
        st.markdown(f"""
        <div class='{css}'>
            <b>{q['id']}</b> — {q['text']}<br>
            <span style='font-size:1.05rem;'>{icon} <b>{label}</b> &nbsp;·&nbsp; Answer: <b>{result['answer']}</b></span><br>
            <div style='margin-top:6px; font-size:0.85rem;'>{metric_html}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    if st.button("▶️ Start ISO Questionnaire →", type="primary"):
        st.session_state.human_answers     = {}
        st.session_state.claude_evals_done = False
        st.session_state.step = 3
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — QUESTIONNAIRE  (free-text answers)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    human_qs  = [q for q in QUESTIONS if q["mode"] == "human"]
    human_ids = [q["id"] for q in human_qs]
    q_by_id   = {q["id"]: q for q in QUESTIONS}

    answered_ids = [qid for qid in human_ids
                    if st.session_state.human_answers.get(qid, "").strip()]
    answered = len(answered_ids)

    st.markdown("### 📋 Step 3 — ISO 24027 Questionnaire")
    st.info(
        "💡 **Answer each question in your own words.** Describe the methods, tools, "
        "or practices your team uses. Claude will evaluate your answer on the report "
        "page and give you a score out of 10 plus written feedback."
    )

    # All answered → proceed
    if answered >= len(human_ids):
        st.success("✅ All questions answered!")
        if st.button("▶️ Generate Bias Report →", type="primary"):
            st.session_state.all_results       = evaluate_all(
                st.session_state.human_answers,
                st.session_state.dbdm_metrics,
            )
            st.session_state.claude_evals_done = False
            st.session_state.step = 4
            st.rerun()
        st.stop()

    st.progress(answered / len(human_ids))
    st.caption(f"**{answered} / {len(human_ids)}** questions answered")

    # Answered summary (collapsible)
    if answered > 0:
        with st.expander(f"Previously answered ({answered})", expanded=False):
            for qid in answered_ids:
                q       = q_by_id[qid]
                ans     = st.session_state.human_answers.get(qid, "")
                preview = ans[:120] + "…" if len(ans) > 120 else ans
                st.markdown(f"""
                <div class='safe-box'>
                    <b>{qid}</b> [{q['category']}]<br>
                    {q['text']}<br>
                    <small><i>✏️ {preview}</i></small>
                </div>
                """, unsafe_allow_html=True)

    # First unanswered question
    unanswered = [qid for qid in human_ids
                  if not st.session_state.human_answers.get(qid, "").strip()]
    if not unanswered:
        st.rerun()
    current_id = unanswered[0]
    current_q  = q_by_id[current_id]
    progress_label = f"Question {human_ids.index(current_id)+1} of {len(human_ids)}"

    st.markdown(f"""
    <div class='q-card'>
        <div style='color:#1a237e; font-size:0.85rem; font-weight:600; margin-bottom:4px;'>
            {current_q['id']} &nbsp;·&nbsp; {current_q['category']}
            &nbsp;·&nbsp; {progress_label}
        </div>
        <div style='font-size:1.1rem; font-weight:500;'>
            {current_q['text']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show relevant DBDM metrics for context
    rel_metrics = CATEGORY_METRICS.get(current_q["category"], [])
    if rel_metrics:
        metric_parts = []
        for m in rel_metrics:
            v = st.session_state.dbdm_metrics.get(m)
            if v is not None:
                flag = "⚠️" if _metric_is_risky(m, v) else "✅"
                metric_parts.append(f"{flag} **{m}** = `{round(float(v), 4)}`")
        if metric_parts:
            st.markdown("**Relevant DBDM metrics:** " + " &nbsp;|&nbsp; ".join(metric_parts))

    # Free-text answer box
    st.markdown("**Your answer:**")
    st.caption(
        "Describe whether this practice is in place, how it is implemented, "
        "and which methods or tools you use. Be specific — Claude will score "
        "the quality and depth of your explanation (0–10)."
    )
    current_draft = st.session_state.human_answers.get(current_id, "")
    new_answer = st.text_area(
        label="Your answer",
        value=current_draft,
        height=140,
        placeholder=(
            "e.g. 'We use stratified sampling to ensure all demographic groups are "
            "represented proportionally. Additionally, we compute Class Imbalance (CI) "
            "before training and apply SMOTE when CI exceeds 0.1…'"
        ),
        key=f"ans_{current_id}",
        label_visibility="collapsed",
    )

    col_submit, col_skip = st.columns([2, 1])
    with col_submit:
        if st.button("✅  Submit Answer →", type="primary",
                     use_container_width=True,
                     disabled=not new_answer.strip()):
            st.session_state.human_answers[current_id] = new_answer.strip()
            st.rerun()
    with col_skip:
        if st.button("⏭️  Skip (mark as not addressed)", use_container_width=True):
            st.session_state.human_answers[current_id] = "[Not addressed]"
            st.rerun()

    # Re-answer previous question
    if answered > 0:
        st.divider()
        st.markdown("**Re-answer a previous question:**")
        prev_map = {
            f"{qid2} — {q_by_id[qid2]['text'][:65]}…": qid2
            for qid2 in answered_ids
        }
        sel = st.selectbox("Select question to change", list(prev_map.keys()),
                           index=None, placeholder="Choose…",
                           label_visibility="collapsed")
        if sel:
            t_qid   = prev_map[sel]
            old_ans = st.session_state.human_answers.get(t_qid, "")
            new_val = st.text_area(
                f"Edit answer for {t_qid}", value=old_ans,
                height=100, key=f"re_{t_qid}"
            )
            if st.button(f"💾 Save updated answer for {t_qid}"):
                st.session_state.human_answers[t_qid] = new_val.strip()
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — BIAS REPORT  (Claude evaluates each human answer)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    results = st.session_state.all_results
    if not results:
        st.warning("No results. Please complete the questionnaire first.")
        st.stop()

    # ── Run Claude evaluations once ───────────────────────────────────────────
    if not st.session_state.claude_evals_done:
        human_results = [r for r in results if r["mode"] == "human"]
        ds    = _dataset_summary()
        total = len(human_results)

        st.markdown("""
        <div class='header-strip'>
            <h2 style='margin:0;color:white;'>🤖 Claude is evaluating your answers…</h2>
            <p style='margin:4px 0 0;color:#b3c5f8;'>This runs once. Results are cached for the session.</p>
        </div>""", unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status_text  = st.empty()

        for i, r in enumerate(human_results):
            qid = r["id"]
            q   = next(qq for qq in QUESTIONS if qq["id"] == qid)
            status_text.caption(f"Evaluating {qid} ({i+1}/{total})…")

            suggestion = generate_suggested_answer(
                q, st.session_state.dbdm_metrics, THRESHOLD_DISPLAY, ds
            )
            evaluation = evaluate_user_answer(
                q, r["user_answer"],
                st.session_state.dbdm_metrics, THRESHOLD_DISPLAY, ds
            )

            r["claude_suggestion"] = suggestion
            r["claude_score"]      = evaluation["score"]
            r["claude_verdict"]    = evaluation["verdict"]
            r["claude_feedback"]   = evaluation["feedback"]
            r["claude_key_gap"]    = evaluation["key_gap"]
            r["is_risky"]          = (evaluation["verdict"] == "Risky")
            r["risk_reason"]       = (
                f"Claude score {evaluation['score']}/10 — {evaluation['feedback']}"
            )

            progress_bar.progress((i + 1) / total)

        status_text.empty()
        progress_bar.empty()
        st.session_state.claude_evals_done = True
        st.rerun()

    # ── Summary header ────────────────────────────────────────────────────────
    risky     = [r for r in results if r["is_risky"]]
    not_risky = [r for r in results if not r["is_risky"]]
    auto_res  = [r for r in results if r["mode"] == "auto"]

    st.markdown("""
    <div class='header-strip'>
        <h2 style='margin:0;color:white;'>📊 AgentFair — Bias Detection Report</h2>
        <p style='margin:4px 0 0;color:#b3c5f8;'>ISO 24027 + DBDM + Claude AI Evaluation</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Questions", len(results))
    c2.metric("⚠️ Risky", len(risky),
              delta=f"{len(risky)/len(results)*100:.0f}% of total", delta_color="inverse")
    c3.metric("✅ Not Risky", len(not_risky))
    c4.metric("🤖 Auto-answered", len(auto_res))

    st.divider()

    # DBDM recap
    with st.expander("📊 DBDM Metrics (Step 2 results)", expanded=False):
        m_rows = []
        for name, val in st.session_state.dbdm_metrics.items():
            risky_m = _metric_is_risky(name, val)
            thr = THRESHOLD_DISPLAY.get(name, "±0.1")
            m_rows.append({
                "Metric": name,
                "Value": round(float(val), 4) if val is not None else "—",
                "Threshold": thr,
                "Status": "⚠️ RISKY" if risky_m else "✅ FAIR",
            })
        mdf2 = pd.DataFrame(m_rows)
        def cs2(row):
            return ["background-color:#fff3cd"
                    if row["Status"].startswith("⚠️")
                    else "background-color:#e8f5e9"] * len(row)
        st.dataframe(mdf2.style.apply(cs2, axis=1),
                     use_container_width=True, hide_index=True)

    st.divider()

    # ── Full results by category ──────────────────────────────────────────────
    st.markdown("## 📋 All Questions — Risk Status")
    categories = list(dict.fromkeys(r["category"] for r in results))

    for cat in categories:
        cat_res  = [r for r in results if r["category"] == cat]
        n_risky  = sum(1 for r in cat_res if r["is_risky"])
        icon_hdr = "⚠️" if n_risky else "✅"

        rel_metrics = CATEGORY_METRICS.get(cat, [])
        metric_snippet = ""
        if rel_metrics:
            parts = []
            for m in rel_metrics:
                v = st.session_state.dbdm_metrics.get(m)
                if v is not None:
                    fl = "⚠️" if _metric_is_risky(m, v) else "✅"
                    parts.append(f"{fl} {m} = {round(float(v),4)}")
            if parts:
                metric_snippet = " &nbsp;|&nbsp; ".join(parts[:4])

        with st.expander(
            f"{icon_hdr} **{cat}** — {n_risky}/{len(cat_res)} risky",
            expanded=(n_risky > 0)
        ):
            if metric_snippet:
                st.markdown(
                    f"<small>Relevant DBDM metrics: {metric_snippet}</small>",
                    unsafe_allow_html=True
                )
                st.markdown("")

            for r in cat_res:
                icon       = "⚠️" if r["is_risky"] else "✅"
                risk_label = "**RISKY**" if r["is_risky"] else "Not Risky"
                css        = "risky-box" if r["is_risky"] else "safe-box"

                if r["mode"] == "auto":
                    metric_html = ""
                    if r.get("metric_values"):
                        parts2 = []
                        for m, v in r["metric_values"].items():
                            thr     = THRESHOLD_DISPLAY.get(m, "±0.1")
                            outside = m in r.get("risky_metrics", {})
                            fl      = "⚠️" if outside else "✅"
                            parts2.append(f"{fl} {m} = {v} (threshold {thr})")
                        metric_html = "<br><small>" + " &nbsp;|&nbsp; ".join(parts2) + "</small>"

                    st.markdown(f"""
                    <div class='{css}'>
                        <span style='font-size:0.8rem;color:#555;'>
                            {r['id']} &nbsp;·&nbsp; 🤖 Auto (DBDM)
                        </span><br>
                        <strong>{r['text']}</strong><br>
                        {icon} {risk_label} &nbsp;·&nbsp; Answer: <b>{r['answer']}</b><br>
                        <small>{r['risk_reason']}</small>
                        {metric_html}
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    score    = r.get("claude_score", "—")
                    chip_cls = "score-chip-risky" if r["is_risky"] else "score-chip-safe"

                    user_ans = r.get("user_answer", "(no answer)")
                    if len(user_ans) > 220:
                        user_ans = user_ans[:220] + "…"

                    suggestion_html = ""
                    if r.get("claude_suggestion"):
                        suggestion_html = f"""
                        <div style='margin-top:10px; background:#f3e5f5;
                             border-left:3px solid #8e24aa; padding:8px 12px;
                             border-radius:4px; font-size:0.88rem;'>
                            <b>🤖 Claude's suggested answer:</b><br>
                            {r['claude_suggestion']}
                        </div>"""

                    gap_html = ""
                    if r.get("claude_key_gap") and r["claude_key_gap"].lower() != "none":
                        gap_html = f"""
                        <div style='margin-top:6px;font-size:0.85rem;color:#b71c1c;'>
                            🔑 <b>Key gap:</b> {r['claude_key_gap']}
                        </div>"""

                    st.markdown(f"""
                    <div class='{css}'>
                        <span style='font-size:0.8rem;color:#555;'>
                            {r['id']} &nbsp;·&nbsp; 👤 Human + 🤖 Claude Evaluation
                        </span><br>
                        <strong>{r['text']}</strong><br>
                        <br>
                        <b>Your answer:</b>
                        <i style='font-size:0.9rem;'>{user_ans}</i><br>
                        <br>
                        {icon} {risk_label} &nbsp;·&nbsp;
                        <span class='{chip_cls}'>{score}/10</span><br>
                        <br>
                        <b>Claude's feedback:</b> {r.get('claude_feedback', '')}
                        {gap_html}
                        {suggestion_html}
                    </div>
                    """, unsafe_allow_html=True)

    st.divider()

    # ── Risky summary ─────────────────────────────────────────────────────────
    st.markdown("## ⚠️ Risky Questions — Quick Reference")
    if not risky:
        st.success("No risky questions detected.")
    else:
        for i, r in enumerate(risky, 1):
            mode_str  = "🤖 Auto" if r["mode"] == "auto" else "👤 Human"
            score_str = f" &nbsp;|&nbsp; Score: **{r['claude_score']}/10**" \
                        if r.get("claude_score") is not None and r["mode"] == "human" else ""
            st.markdown(
                f"**{i}. {r['id']} [{r['category']}]** ({mode_str}){score_str}  \n"
                f"📌 {r['text']}  \n"
                f"*{r['risk_reason']}*"
            )
            st.markdown("---")

    st.divider()

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("## 💾 Download Report")
    report = {
        "dataset_info": {
            "facet":       st.session_state.facet,
            "outcome":     st.session_state.outcome,
            "subgroup":    st.session_state.subgroup,
            "label_value": st.session_state.label_value,
            "rows": len(st.session_state.df) if st.session_state.df is not None else "N/A",
        },
        "dbdm_metrics": {
            k: {
                "value": v,
                "threshold": THRESHOLD_DISPLAY.get(k, "±0.1"),
                "is_risky": _metric_is_risky(k, v),
            }
            for k, v in st.session_state.dbdm_metrics.items()
        },
        "summary": {
            "total": len(results), "risky": len(risky),
            "not_risky": len(not_risky), "auto_answered": len(auto_res),
        },
        "iso_questionnaire": [
            {
                "id": r["id"], "category": r["category"], "mode": r["mode"],
                "text": r["text"],
                "user_answer": r.get("user_answer", ""),
                "is_risky": r["is_risky"],
                "claude_score":      r.get("claude_score"),
                "claude_verdict":    r.get("claude_verdict", ""),
                "claude_feedback":   r.get("claude_feedback", ""),
                "claude_key_gap":    r.get("claude_key_gap", ""),
                "claude_suggestion": r.get("claude_suggestion", ""),
                "dbdm_metric_values": r.get("metric_values", {}),
            }
            for r in results
        ],
    }

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "📥 Download Full Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="agentfair_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with d2:
        csv_rows = [{
            "ID": r["id"], "Category": r["category"], "Mode": r["mode"],
            "Question": r["text"],
            "User Answer": r.get("user_answer", ""),
            "Is Risky": r["is_risky"],
            "Claude Score":      r.get("claude_score", ""),
            "Claude Verdict":    r.get("claude_verdict", ""),
            "Claude Feedback":   r.get("claude_feedback", ""),
            "Claude Key Gap":    r.get("claude_key_gap", ""),
            "Claude Suggestion": r.get("claude_suggestion", ""),
        } for r in results]
        st.download_button(
            "📥 Download Questions (CSV)",
            data=pd.DataFrame(csv_rows).to_csv(index=False),
            file_name="agentfair_questions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()
    if st.button("🔄 Run Again"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
