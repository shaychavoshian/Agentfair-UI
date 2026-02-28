# app.py  –  AgentFair Interactive Bias Detection UI
# Run:  streamlit run app.py

import streamlit as st
import pandas as pd
import tempfile
import os
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi

sys.path.insert(0, os.path.dirname(__file__))

from core.questionnaire import (
    QUESTIONS, DBDM_THRESHOLDS, THRESHOLD_DISPLAY,
    CATEGORY_METRICS, evaluate_all, auto_answer, _metric_is_risky,
)
from core.dbdm_runner import run_dbdm
from core.claude_evaluator import generate_suggested_answer, evaluate_user_answer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgentFair – ISO 24027 Bias Detection",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: works in both light AND dark mode ──────────────────────────────────
st.markdown("""
<style>
.header-strip {
    background: linear-gradient(90deg,#1a237e,#283593);
    color: #ffffff !important;
    padding: 16px 24px; border-radius: 8px; margin-bottom: 20px;
}
.header-strip h2, .header-strip p { color: #ffffff !important; }
.q-card {
    background: #e8eaf6;
    border: 2px solid #c5cae9;
    border-radius: 10px; padding: 20px 24px; margin: 12px 0;
    color: #1a1a2e !important;
}
.q-card * { color: #1a1a2e !important; }
.risky-box {
    background: #fff3cd; border-left: 4px solid #e53935;
    padding: 12px 16px; border-radius: 6px; margin-bottom: 8px;
    color: #1a1a1a !important;
}
.risky-box * { color: #1a1a1a !important; }
.safe-box {
    background: #e8f5e9; border-left: 4px solid #43a047;
    padding: 12px 16px; border-radius: 6px; margin-bottom: 8px;
    color: #1a1a1a !important;
}
.safe-box * { color: #1a1a1a !important; }
.auto-box {
    background: #e3f2fd; border-left: 4px solid #1e88e5;
    padding: 12px 16px; border-radius: 6px; margin-bottom: 8px;
    color: #1a1a1a !important;
}
.auto-box * { color: #1a1a1a !important; }
.score-chip-risky {
    display: inline-block; background: #e53935; color: #fff !important;
    border-radius: 12px; padding: 2px 12px; font-weight: 700;
}
.score-chip-safe {
    display: inline-block; background: #43a047; color: #fff !important;
    border-radius: 12px; padding: 2px 12px; font-weight: 700;
}
.yn-yes {
    background: #e8f5e9; border: 2px solid #43a047; border-radius: 8px;
    padding: 4px 14px; color: #1b5e20 !important; font-weight: 700;
    display: inline-block; margin: 4px 0;
}
.yn-no {
    background: #ffebee; border: 2px solid #e53935; border-radius: 8px;
    padding: 4px 14px; color: #b71c1c !important; font-weight: 700;
    display: inline-block; margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "step": 1, "df": None, "csv_path": None,
        "facet": None, "outcome": None, "subgroup": "",
        "label_value": 1.0, "dbdm_metrics": {}, "dbdm_risk_levels": {},
        "dbdm_json_path": None, "human_answers": {},
        "all_results": [], "claude_evals_done": False, "dbdm_error": None,
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

def _get_yn(qid):
    raw = st.session_state.human_answers.get(qid, {})
    if isinstance(raw, dict): return raw.get("yn", "")
    return ""

def _get_desc(qid):
    raw = st.session_state.human_answers.get(qid, {})
    if isinstance(raw, dict): return raw.get("desc", "")
    return raw if isinstance(raw, str) else ""

def _answer_text(qid):
    yn = _get_yn(qid); desc = _get_desc(qid).strip()
    if not yn: return ""
    return f"{yn}. {desc}" if desc else yn

def _is_answered(qid):
    raw = st.session_state.human_answers.get(qid, "")
    if raw == "[Not addressed]": return True
    return bool(_get_yn(qid))


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ AgentFair")
    st.markdown("*ISO 24027 + DBDM + Claude AI*")
    st.divider()
    for num, label in [(1,"Upload Dataset"),(2,"DBDM Metrics"),
                        (3,"ISO Questionnaire"),(4,"Bias Report")]:
        cur = st.session_state.step == num
        done = st.session_state.step > num
        icon = "✅" if done else ("▶️" if cur else "○")
        txt = f"**Step {num}: {label}**" if cur else f"Step {num}: {label}"
        st.markdown(f"{icon} {txt}")
    if st.session_state.step >= 3:
        human_qs = [q for q in QUESTIONS if q["mode"] == "human"]
        answered = sum(1 for q in human_qs if _is_answered(q["id"]))
        st.divider()
        st.markdown("**Questionnaire progress**")
        st.progress(answered / max(len(human_qs), 1))
        st.caption(f"{answered} / {len(human_qs)} answered")
    if st.session_state.step > 1:
        st.divider()
        if st.button("🔄 Start Over", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.markdown("""
    <div class='header-strip'>
        <h2 style='margin:0;'>⚖️ AgentFair — AI Bias Detection</h2>
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
            facet = st.selectbox("Sensitive attribute (facet)", cols)
        with c2:
            outcome = st.selectbox("Outcome column", [c for c in cols if c != facet])
        with c3:
            subgroup = st.selectbox("Subgroup column (optional)",
                ["(none)"] + [c for c in cols if c not in [facet, outcome]])
        label_value = st.number_input("Positive label value", value=1.0, step=0.5)
        if st.button("▶️ Continue →", type="primary"):
            st.session_state.facet = facet
            st.session_state.outcome = outcome
            st.session_state.subgroup = "" if subgroup == "(none)" else subgroup
            st.session_state.label_value = label_value
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            df.to_csv(tmp.name, index=False)
            st.session_state.csv_path = tmp.name
            st.session_state.step = 2
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DBDM METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    st.markdown("### ⚙️ Step 2 — DBDM Bias Metrics")
    if not st.session_state.dbdm_metrics and not st.session_state.dbdm_error:
        with st.spinner("Running DBDM — calculating fairness metrics…"):
            try:
                metrics, risk_levels, json_path = run_dbdm(
                    file_path=st.session_state.csv_path,
                    facet=st.session_state.facet,
                    outcome=st.session_state.outcome,
                    subgroup_col=st.session_state.subgroup,
                    label_value=st.session_state.label_value,
                )
                st.session_state.dbdm_metrics = metrics
                st.session_state.dbdm_risk_levels = risk_levels
                st.session_state.dbdm_json_path = json_path
            except Exception as e:
                st.session_state.dbdm_error = str(e)
    if st.session_state.dbdm_error:
        st.error(f"❌ DBDM failed: {st.session_state.dbdm_error}")
        if st.button("← Back"):
            st.session_state.step = 1; st.session_state.dbdm_error = None; st.rerun()
        st.stop()

    metrics = st.session_state.dbdm_metrics
    st.success("✅ Metrics calculated successfully.")

    # ── Metrics table
    rows = []
    for name, val in metrics.items():
        risky = _metric_is_risky(name, val)
        rows.append({
            "Metric": name,
            "Value": round(float(val), 4) if val is not None else "—",
            "Threshold": THRESHOLD_DISPLAY.get(name, "±0.1"),
            "DBDM Risk Level": st.session_state.dbdm_risk_levels.get(name, "—"),
            "Status": "⚠️ RISKY" if risky else "✅ FAIR",
        })
    mdf = pd.DataFrame(rows)
    def color_status(row):
        return (["background-color:#fff3cd"] if row["Status"].startswith("⚠️")
                else ["background-color:#e8f5e9"]) * len(row)
    st.dataframe(mdf.style.apply(color_status, axis=1), use_container_width=True, hide_index=True)
    rc = sum(1 for r in rows if r["Status"].startswith("⚠️"))
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Metrics", len(rows)); c2.metric("⚠️ Outside Threshold", rc); c3.metric("✅ Within Threshold", len(rows)-rc)

    # ── DBDM bar plots with red/green zones
    st.divider()
    st.markdown("#### 📊 DBDM Metric Plots")
    METRIC_ORDER = [
        "Class Imbalance (CI)", "Difference in Proportion Labels (DPL)",
        "Demographic Disparity (DD)", "Jensen-Shannon Divergence (JS)", "L2 Norm",
        "KS value", "Normalized Mutual Information (NMI)", "Binary Ratio (BR)",
        "Binary Difference (BD)", "Pearson Correlation (CORR)",
        "Total Variation Distance (TVD)", "Conditional Demographic Disparity (CDD)",
        "Normalized Conditional Mutual Information (NCMI)",
        "Conditional Binary Difference (CBD)", "Logistic Regression Coefficient (LR)",
    ]
    plot_m = {k: metrics[k] for k in METRIC_ORDER if k in metrics}
    n = len(plot_m); ncols = 5; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3*nrows))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    lr = "#f08080"; lg = "#8fbc8f"
    for i, (name, val) in enumerate(plot_m.items()):
        ax = axes_flat[i]
        if val is not None:
            xv = float(val)
            ax.barh([""], [xv], color="grey", height=0.5)
            ax.text(xv+(0.05 if xv>=0 else -0.05), 0, f"{xv:.3f}",
                    ha="left" if xv>=0 else "right", va="center", fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        thr = DBDM_THRESHOLDS.get(name, 0.1)
        if name == "Binary Ratio (BR)":
            ax.set_xlim(-0.5, 2.5)
            ax.axvspan(-0.5, 0.8, facecolor=lr, alpha=0.4)
            ax.axvspan(0.8, 1.25, facecolor=lg, alpha=0.4)
            ax.axvspan(1.25, 2.5, facecolor=lr, alpha=0.4)
        elif name == "Logistic Regression Coefficient (LR)":
            ax.set_xlim(-3, 3)
            ax.axvspan(-3,-2,facecolor=lr,alpha=0.4); ax.axvspan(-2,2,facecolor=lg,alpha=0.4); ax.axvspan(2,3,facecolor=lr,alpha=0.4)
        else:
            t = thr if isinstance(thr, float) else 0.1
            ax.set_xlim(-1, 1)
            ax.axvspan(-1,-t,facecolor=lr,alpha=0.4); ax.axvspan(-t,t,facecolor=lg,alpha=0.4); ax.axvspan(t,1,facecolor=lr,alpha=0.4)
        ax.set_title(name, fontsize=8); ax.set_yticks([]); ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
        if i == 0:
            ax.legend(handles=[mpatches.Patch(color=lg,alpha=0.7,label="Fair"),
                                mpatches.Patch(color=lr,alpha=0.7,label="Bias")], fontsize=7, loc="upper right")
    for j in range(n, len(axes_flat)): axes_flat[j].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Auto-answered questions
    st.divider()
    st.markdown("#### Auto-answered Questions (Q35, Q36, Q37)")
    for q in [q for q in QUESTIONS if q["mode"] == "auto"]:
        result = auto_answer(q, metrics)
        icon = "⚠️" if result["is_risky"] else "✅"
        css = "risky-box" if result["is_risky"] else "auto-box"
        mparts = []
        for m, v in result["values"].items():
            thr = THRESHOLD_DISPLAY.get(m, "±0.1")
            fl = "⚠️" if m in result["risky_metrics"] else "✅"
            mparts.append(f"{fl}&nbsp;<b>{m}</b> = {v}&nbsp;(threshold {thr})")
        st.markdown(f"""
        <div class='{css}'>
            <b>{q['id']}</b> — {q['text']}<br>
            {icon} <b>{'RISKY' if result['is_risky'] else 'NOT RISKY'}</b>
            &nbsp;·&nbsp; Answer: <b>{result['answer']}</b><br>
            <div style='margin-top:6px;font-size:0.85rem;'>{'<br>'.join(mparts)}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    if st.button("▶️ Start ISO Questionnaire →", type="primary"):
        st.session_state.human_answers = {}
        st.session_state.claude_evals_done = False
        st.session_state.step = 3
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — QUESTIONNAIRE  (Yes/No first, then optional description)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    human_qs = [q for q in QUESTIONS if q["mode"] == "human"]
    human_ids = [q["id"] for q in human_qs]
    q_by_id = {q["id"]: q for q in QUESTIONS}
    answered_ids = [qid for qid in human_ids if _is_answered(qid)]
    answered = len(answered_ids)

    st.markdown("### 📋 Step 3 — ISO 24027 Questionnaire")
    st.info("💡 **For each question: choose Yes or No first, then optionally add details.** Claude will score your answer 0–10.")

    if answered >= len(human_ids):
        st.success("✅ All questions answered!")
        if st.button("▶️ Generate Bias Report →", type="primary"):
            plain = {qid: _answer_text(qid) for qid in human_ids}
            st.session_state.all_results = evaluate_all(plain, st.session_state.dbdm_metrics)
            st.session_state.claude_evals_done = False
            st.session_state.step = 4
            st.rerun()
        st.stop()

    st.progress(answered / len(human_ids))
    st.caption(f"**{answered} / {len(human_ids)}** answered")

    if answered > 0:
        with st.expander(f"Previously answered ({answered})", expanded=False):
            for qid in answered_ids:
                q = q_by_id[qid]
                raw = st.session_state.human_answers.get(qid, "")
                if raw == "[Not addressed]":
                    yn_html = "<span class='yn-no'>Skipped</span>"
                else:
                    yn = _get_yn(qid); desc = _get_desc(qid)
                    yn_css = "yn-yes" if yn == "Yes" else "yn-no"
                    preview = (desc[:90]+"…" if len(desc)>90 else desc) if desc else ""
                    yn_html = f"<span class='{yn_css}'>{yn}</span>" + (f"<br><small><i>{preview}</i></small>" if preview else "")
                st.markdown(f"<div class='safe-box'><b>{qid}</b> [{q['category']}]<br>{q['text']}<br>{yn_html}</div>",
                            unsafe_allow_html=True)

    unanswered = [qid for qid in human_ids if not _is_answered(qid)]
    if not unanswered: st.rerun()
    cur_id = unanswered[0]
    cur_q = q_by_id[cur_id]
    progress_label = f"Question {human_ids.index(cur_id)+1} of {len(human_ids)}"
    cur_yn = _get_yn(cur_id)
    cur_desc = _get_desc(cur_id)

    st.markdown(f"""
    <div class='q-card'>
        <div style='font-size:0.85rem;font-weight:600;margin-bottom:6px;'>
            {cur_q['id']} &nbsp;·&nbsp; {cur_q['category']} &nbsp;·&nbsp; {progress_label}
        </div>
        <div style='font-size:1.1rem;font-weight:500;'>{cur_q['text']}</div>
    </div>""", unsafe_allow_html=True)

    rel = CATEGORY_METRICS.get(cur_q["category"], [])
    if rel:
        parts = []
        for m in rel:
            v = st.session_state.dbdm_metrics.get(m)
            if v is not None:
                fl = "⚠️" if _metric_is_risky(m, v) else "✅"
                parts.append(f"{fl} **{m}** = `{round(float(v),4)}`")
        if parts:
            st.markdown("**Relevant DBDM metrics:** " + " &nbsp;|&nbsp; ".join(parts))

    # ── YES / NO buttons ──────────────────────────────────────────────────────
    st.markdown("**Step 1 — Your answer:**")
    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        if st.button("✅  Yes", type="primary" if cur_yn=="Yes" else "secondary",
                     use_container_width=True, key=f"yes_{cur_id}"):
            st.session_state.human_answers[cur_id] = {"yn":"Yes","desc":cur_desc}; st.rerun()
    with b2:
        if st.button("❌  No", type="primary" if cur_yn=="No" else "secondary",
                     use_container_width=True, key=f"no_{cur_id}"):
            st.session_state.human_answers[cur_id] = {"yn":"No","desc":cur_desc}; st.rerun()
    with b3:
        if st.button("⏭️  Skip", use_container_width=True, key=f"skip_{cur_id}"):
            st.session_state.human_answers[cur_id] = "[Not addressed]"; st.rerun()

    if cur_yn:
        yn_css = "yn-yes" if cur_yn == "Yes" else "yn-no"
        st.markdown(f"<span class='{yn_css}'>Selected: {cur_yn}</span>", unsafe_allow_html=True)

    # ── Optional description ──────────────────────────────────────────────────
    desc_label = "**Step 2 — Additional details** *(optional)*:"
    if cur_q.get("q_type") == "describe" and cur_yn == "Yes":
        desc_label = "**Step 2 — Please describe** *(required for Yes answers to this question)*:"
    st.markdown(desc_label)
    st.caption(cur_q.get("comment_prompt", "Describe methods, tools, or practices in place."))
    new_desc = st.text_area("desc", value=cur_desc, height=120,
        placeholder="e.g. 'We apply stratified sampling; CI is checked before training…'",
        key=f"desc_{cur_id}", label_visibility="collapsed")

    st.markdown("")
    if st.button("✅  Submit & Next →", type="primary", disabled=not cur_yn, key=f"sub_{cur_id}"):
        st.session_state.human_answers[cur_id] = {"yn": cur_yn, "desc": new_desc.strip()}
        st.rerun()
    if not cur_yn:
        st.caption("⬆️ Choose Yes or No above before submitting.")

    if answered > 0:
        st.divider()
        st.markdown("**Re-answer a previous question:**")
        prev_map = {f"{qid2} — {q_by_id[qid2]['text'][:65]}…": qid2 for qid2 in answered_ids}
        sel = st.selectbox("", list(prev_map.keys()), index=None, placeholder="Choose…", label_visibility="collapsed")
        if sel:
            t_qid = prev_map[sel]
            old_yn = _get_yn(t_qid); old_desc = _get_desc(t_qid)
            r1, r2 = st.columns(2)
            with r1:
                if st.button("✅ Yes", key=f"ry_{t_qid}"):
                    st.session_state.human_answers[t_qid] = {"yn":"Yes","desc":old_desc}; st.rerun()
            with r2:
                if st.button("❌ No", key=f"rn_{t_qid}"):
                    st.session_state.human_answers[t_qid] = {"yn":"No","desc":old_desc}; st.rerun()
            new_rd = st.text_area(f"Edit description for {t_qid}", value=old_desc, height=80, key=f"rd_{t_qid}")
            if st.button(f"💾 Save {t_qid}"):
                st.session_state.human_answers[t_qid] = {"yn":old_yn,"desc":new_rd.strip()}; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — BIAS REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    results = st.session_state.all_results
    if not results:
        st.warning("No results. Please complete the questionnaire first."); st.stop()

    # ── Claude evaluations (with graceful 401 fallback) ──────────────────────
    if not st.session_state.claude_evals_done:
        human_results = [r for r in results if r["mode"] == "human"]
        ds = _dataset_summary()
        total = len(human_results)
        st.markdown("""
        <div class='header-strip'>
            <h2 style='margin:0;'>🤖 Claude is evaluating your answers…</h2>
            <p style='margin:4px 0 0;color:#b3c5f8;'>Results cached for this session.</p>
        </div>""", unsafe_allow_html=True)
        pb = st.progress(0); stxt = st.empty()

        for i, r in enumerate(human_results):
            qid = r["id"]
            q = next(qq for qq in QUESTIONS if qq["id"] == qid)
            stxt.caption(f"Evaluating {qid} ({i+1}/{total})…")

            try:
                suggestion = generate_suggested_answer(q, st.session_state.dbdm_metrics, THRESHOLD_DISPLAY, ds)
            except Exception:
                suggestion = "(Claude API unavailable)"

            try:
                evaluation = evaluate_user_answer(q, r["user_answer"], st.session_state.dbdm_metrics, THRESHOLD_DISPLAY, ds)
            except Exception:
                yn = _get_yn(qid)
                risky_ans = (yn == "No" and q.get("risky_on") == "No") or (yn == "Yes" and q.get("risky_on") == "Yes")
                evaluation = {
                    "score": 3 if risky_ans else 7,
                    "verdict": "Risky" if risky_ans else "Not Risky",
                    "feedback": "Claude API unavailable. Score estimated from Yes/No response.",
                    "key_gap": "Re-run when API is configured.",
                }

            r["claude_suggestion"] = suggestion
            r["claude_score"] = evaluation["score"]
            r["claude_verdict"] = evaluation["verdict"]
            r["claude_feedback"] = evaluation["feedback"]
            r["claude_key_gap"] = evaluation["key_gap"]
            r["is_risky"] = (evaluation["verdict"] == "Risky")
            r["risk_reason"] = f"Claude score {evaluation['score']}/10 — {evaluation['feedback']}"
            pb.progress((i+1)/total)

        stxt.empty(); pb.empty()
        st.session_state.claude_evals_done = True
        st.rerun()

    risky = [r for r in results if r["is_risky"]]
    not_risky = [r for r in results if not r["is_risky"]]
    auto_res = [r for r in results if r["mode"] == "auto"]

    st.markdown("""
    <div class='header-strip'>
        <h2 style='margin:0;'>📊 AgentFair — Bias Detection Report</h2>
        <p style='margin:4px 0 0;color:#b3c5f8;'>ISO 24027 + DBDM + Claude AI Evaluation</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Questions", len(results))
    c2.metric("⚠️ Risky", len(risky), delta=f"{len(risky)/len(results)*100:.0f}%", delta_color="inverse")
    c3.metric("✅ Not Risky", len(not_risky))
    c4.metric("🤖 Auto-answered", len(auto_res))

    st.divider()

    # ── Spider / Radar plot ───────────────────────────────────────────────────
    st.markdown("#### 🕸️ Bias Profile — ISO 24027 Categories")
    from math import pi as _pi
    cats = list(CATEGORY_METRICS.keys())
    scores = []
    for cat in cats:
        cr = [r for r in results if r.get("category") == cat]
        scores.append(sum(1 for r in cr if r.get("is_risky",False)) / len(cr) if cr else 0.0)

    N = len(cats)
    angles = [n/float(N)*2*_pi for n in range(N)] + [0]
    sp = scores + scores[:1]
    short = ["Missing\nFeatures","Data Labels\n& Labelling","Non-repr.\nSampling",
             "Selection\nBias","Data\nProcessing","Simpson's\nParadox",
             "Data\nAggregation","Distributed\nTraining","Other\nBias"]
    short = (short + cats)[:N]

    fig2, ax2 = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    ax2.set_facecolor("#f8f9ff"); fig2.patch.set_facecolor("#f8f9ff")
    for rv in [0.25,0.5,0.75,1.0]:
        ax2.plot(angles,[rv]*(N+1),color="#cccccc",linewidth=0.5,linestyle="--")
    ax2.fill(angles, sp, color="#e53935", alpha=0.25)
    ax2.plot(angles, sp, color="#e53935", linewidth=2.5)
    ax2.scatter(angles[:-1], scores, color="#b71c1c", s=60, zorder=5)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(short, fontsize=9, color="#1a237e", fontweight="bold")
    ax2.set_yticks([0.25,0.5,0.75,1.0])
    ax2.set_yticklabels(["25%","50%","75%","100%"], fontsize=7, color="#888")
    ax2.set_ylim(0,1)
    ax2.set_title("Bias Risk by ISO 24027 Category", fontsize=11, pad=20, color="#1a237e", fontweight="bold")

    col_sp, col_leg = st.columns([2,1])
    with col_sp: st.pyplot(fig2)
    with col_leg:
        st.markdown("**Category Risk Scores**")
        for cat, sc in zip(cats, scores):
            bw = int(sc*20)
            col = "🔴" if sc>0.5 else ("🟡" if sc>0.2 else "🟢")
            st.markdown(f"{col} **{cat}**  \n`{'█'*bw}{'░'*(20-bw)}` {sc*100:.0f}%")
    plt.close(fig2)

    st.divider()

    with st.expander("📊 DBDM Metrics (recap)", expanded=False):
        mr = []
        for name, val in st.session_state.dbdm_metrics.items():
            rm = _metric_is_risky(name, val)
            mr.append({"Metric":name,"Value":round(float(val),4) if val is not None else "—",
                       "Threshold":THRESHOLD_DISPLAY.get(name,"±0.1"),
                       "DBDM Risk Level":st.session_state.dbdm_risk_levels.get(name,"—"),
                       "Status":"⚠️ RISKY" if rm else "✅ FAIR"})
        mdf2 = pd.DataFrame(mr)
        def cs2(row): return (["background-color:#fff3cd"] if row["Status"].startswith("⚠️") else ["background-color:#e8f5e9"])*len(row)
        st.dataframe(mdf2.style.apply(cs2,axis=1), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("## 📋 All Questions — Risk Status")
    categories = list(dict.fromkeys(r["category"] for r in results))

    for cat in categories:
        cr = [r for r in results if r["category"] == cat]
        nr = sum(1 for r in cr if r["is_risky"])
        rel_m = CATEGORY_METRICS.get(cat, [])
        msnip = ""
        if rel_m:
            pts = []
            for m in rel_m:
                v = st.session_state.dbdm_metrics.get(m)
                if v is not None:
                    fl = "⚠️" if _metric_is_risky(m,v) else "✅"
                    pts.append(f"{fl} {m}={round(float(v),4)}")
            if pts: msnip = " | ".join(pts[:4])

        with st.expander(f"{'⚠️' if nr else '✅'} **{cat}** — {nr}/{len(cr)} risky", expanded=(nr>0)):
            if msnip:
                st.markdown(f"<small>DBDM: {msnip}</small>", unsafe_allow_html=True)

            for r in cr:
                icon = "⚠️" if r["is_risky"] else "✅"
                rl = "**RISKY**" if r["is_risky"] else "Not Risky"
                css = "risky-box" if r["is_risky"] else "safe-box"

                if r["mode"] == "auto":
                    mh = ""
                    if r.get("metric_values"):
                        p2 = []
                        for m,v in r["metric_values"].items():
                            fl = "⚠️" if m in r.get("risky_metrics",{}) else "✅"
                            p2.append(f"{fl} {m}={v} (thr {THRESHOLD_DISPLAY.get(m,'±0.1')})")
                        mh = "<br><small>" + " | ".join(p2) + "</small>"
                    st.markdown(f"<div class='{css}'><span style='font-size:0.8rem;'>{r['id']} · 🤖 Auto</span><br><strong>{r['text']}</strong><br>{icon} {rl} · Answer: <b>{r['answer']}</b><br><small>{r['risk_reason']}</small>{mh}</div>", unsafe_allow_html=True)
                else:
                    score = r.get("claude_score","—")
                    chip = "score-chip-risky" if r["is_risky"] else "score-chip-safe"
                    yn = _get_yn(r["id"])
                    yn_badge = f"<span class='{'yn-yes' if yn=='Yes' else 'yn-no'}'>{yn}</span>&nbsp;" if yn else ""
                    ua = r.get("user_answer","(no answer)")
                    if len(ua)>220: ua=ua[:220]+"…"
                    sug_html = f"<div style='margin-top:10px;background:#f3e5f5;border-left:3px solid #8e24aa;padding:8px 12px;border-radius:4px;font-size:0.88rem;color:#1a1a1a;'><b>🤖 Claude suggestion:</b><br>{r['claude_suggestion']}</div>" if r.get("claude_suggestion") else ""
                    gap_html = f"<div style='margin-top:6px;font-size:0.85rem;color:#b71c1c;'>🔑 <b>Key gap:</b> {r['claude_key_gap']}</div>" if r.get("claude_key_gap","").lower() not in ["none",""] else ""
                    st.markdown(f"<div class='{css}'><span style='font-size:0.8rem;'>{r['id']} · 👤 Human + 🤖 Claude</span><br><strong>{r['text']}</strong><br><br>{yn_badge}<b>Description:</b> <i style='font-size:0.9rem;'>{ua}</i><br><br>{icon} {rl} &nbsp;·&nbsp; <span class='{chip}'>{score}/10</span><br><br><b>Feedback:</b> {r.get('claude_feedback','')}{gap_html}{sug_html}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("## ⚠️ Risky Questions — Quick Reference")
    if not risky:
        st.success("No risky questions detected.")
    else:
        for i, r in enumerate(risky, 1):
            ms = f" | Score: **{r['claude_score']}/10**" if r.get("claude_score") is not None and r["mode"]=="human" else ""
            st.markdown(f"**{i}. {r['id']} [{r['category']}]** ({'🤖 Auto' if r['mode']=='auto' else '👤 Human'}){ms}  \n📌 {r['text']}  \n*{r['risk_reason']}*")
            st.markdown("---")

    st.divider()
    st.markdown("## 💾 Download Report")
    report = {
        "dataset_info": {"facet":st.session_state.facet,"outcome":st.session_state.outcome,
                         "subgroup":st.session_state.subgroup,"label_value":st.session_state.label_value,
                         "rows":len(st.session_state.df) if st.session_state.df is not None else "N/A"},
        "dbdm_metrics": {k:{"value":v,"threshold":THRESHOLD_DISPLAY.get(k,"±0.1"),
                             "is_risky":_metric_is_risky(k,v),
                             "risk_level":st.session_state.dbdm_risk_levels.get(k,"—")}
                         for k,v in st.session_state.dbdm_metrics.items()},
        "summary": {"total":len(results),"risky":len(risky),"not_risky":len(not_risky),"auto_answered":len(auto_res)},
        "iso_questionnaire": [{"id":r["id"],"category":r["category"],"mode":r["mode"],"text":r["text"],
                                "user_answer":r.get("user_answer",""),"is_risky":r["is_risky"],
                                "claude_score":r.get("claude_score"),"claude_verdict":r.get("claude_verdict",""),
                                "claude_feedback":r.get("claude_feedback",""),"claude_key_gap":r.get("claude_key_gap",""),
                                "claude_suggestion":r.get("claude_suggestion","")} for r in results],
    }
    d1,d2 = st.columns(2)
    with d1:
        st.download_button("📥 Download Full Report (JSON)", data=json.dumps(report,indent=2),
                           file_name="agentfair_report.json", mime="application/json", use_container_width=True)
    with d2:
        csv_rows = [{"ID":r["id"],"Category":r["category"],"Mode":r["mode"],"Question":r["text"],
                     "User Answer":r.get("user_answer",""),"Is Risky":r["is_risky"],
                     "Claude Score":r.get("claude_score",""),"Claude Verdict":r.get("claude_verdict",""),
                     "Claude Feedback":r.get("claude_feedback",""),"Claude Key Gap":r.get("claude_key_gap",""),
                     "Claude Suggestion":r.get("claude_suggestion","")} for r in results]
        st.download_button("📥 Download Questions (CSV)", data=pd.DataFrame(csv_rows).to_csv(index=False),
                           file_name="agentfair_questions.csv", mime="text/csv", use_container_width=True)
    st.divider()
    if st.button("🔄 Run Again"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()
