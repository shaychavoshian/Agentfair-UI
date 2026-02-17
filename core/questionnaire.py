# core/questionnaire.py
"""
All 39 ISO 24027 questions (supplementary material of the book chapter).
Q35, Q36, Q37 are auto-answered from DBDM metrics.
All others: Yes/No + optional comment.

RISK LOGIC
──────────
q_type = "confirm"  →  risk comes from Yes/No only (comment is optional context)
q_type = "describe" →  risk from Yes/No first; if "safe" answer given:
                          comment empty          → RISKY (unsubstantiated claim)
                          comment filled, ≥1 keyword → NOT RISKY
                          comment filled, 0 keywords  → RISKY (vague description)
"""

# ── Thresholds (mirror DBDM.py exactly) ──────────────────────────────────────
DBDM_THRESHOLDS = {
    'Class Imbalance (CI)': 0.1,
    'Difference in Proportion Labels (DPL)': 0.1,
    'Demographic Disparity (DD)': 0.1,
    'Jensen-Shannon Divergence (JS)': 0.1,
    'L2 Norm': 0.1,
    'KS value': 0.1,
    'Normalized Mutual Information (NMI)': 0.1,
    'Binary Ratio (BR)': (0.8, 1.25),
    'Binary Difference (BD)': 0.1,
    'Pearson Correlation (CORR)': 0.1,
    'Total Variation Distance (TVD)': 0.1,
    'Conditional Demographic Disparity (CDD)': 0.1,
    'Normalized Conditional Mutual Information (NCMI)': 0.1,
    'Conditional Binary Difference (CBD)': 0.1,
    'Logistic Regression Coefficient (LR)': 0.1,
    'Logistic Regression Intercept (Intercept)': (-2, 2),
}

THRESHOLD_DISPLAY = {
    k: (f"{v[0]} – {v[1]}" if isinstance(v, tuple) else f"±{v}")
    for k, v in DBDM_THRESHOLDS.items()
}

# Which DBDM metrics are relevant to each ISO category (for Step 2 display)
CATEGORY_METRICS = {
    "Missing features and labels": ["Normalized Mutual Information (NMI)", "Normalized Conditional Mutual Information (NCMI)"],
    "Data labels and labelling process": ["Difference in Proportion Labels (DPL)", "Demographic Disparity (DD)", "Conditional Demographic Disparity (CDD)"],
    "Non-representative sampling": ["Class Imbalance (CI)", "Binary Ratio (BR)", "Binary Difference (BD)", "Conditional Binary Difference (CBD)"],
    "Selection bias": ["Class Imbalance (CI)", "Binary Ratio (BR)", "Binary Difference (BD)", "KS value", "Total Variation Distance (TVD)", "Jensen-Shannon Divergence (JS)", "Pearson Correlation (CORR)", "Normalized Mutual Information (NMI)"],
    "Data processing": ["Demographic Disparity (DD)", "Difference in Proportion Labels (DPL)"],
    "Simpson's paradox": ["Conditional Demographic Disparity (CDD)", "Normalized Conditional Mutual Information (NCMI)", "Conditional Binary Difference (CBD)"],
    "Data aggregation": ["Conditional Demographic Disparity (CDD)", "Normalized Conditional Mutual Information (NCMI)"],
    "Distributed training": [],
    "Other sources of data bias": ["Class Imbalance (CI)", "Demographic Disparity (DD)"],
}

# ── 39 Questions ──────────────────────────────────────────────────────────────
QUESTIONS = [
    # ── Missing features and labels ─────────────────────────────────────────
    {"id": "Q01", "text": "Do the labels adequately represent the diversity of the real-world population or context?",
     "category": "Missing features and labels", "mode": "human", "q_type": "confirm", "risky_on": "No",
     "comment_prompt": "Optional: describe your label coverage approach.", "keywords": []},

    {"id": "Q02", "text": "Are the proxies used for ground truth validation sufficiently representative of the true labels?",
     "category": "Missing features and labels", "mode": "human", "q_type": "confirm", "risky_on": "No",
     "comment_prompt": "Optional: what proxies are used?", "keywords": []},

    {"id": "Q03", "text": "Do you assess potential inaccuracies introduced by the proxies?",
     "category": "Missing features and labels", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe how you assess proxy inaccuracies (e.g. inter-rater agreement, validation study, error analysis).",
     "keywords": ["inter-rater", "kappa", "agreement", "validation", "error analysis", "ground truth",
                  "sensitivity", "specificity", "audit", "review", "cross-valid", "benchmark", "comparison", "evaluat"]},

    {"id": "Q04", "text": "Do you train labelers to recognize and reduce cognitive or societal biases in their work?",
     "category": "Data labels and labelling process", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the training approach (e.g. bias workshops, annotation guidelines, calibration sessions).",
     "keywords": ["workshop", "training", "guideline", "calibr", "instruct", "onboard",
                  "bias aware", "annotation guide", "rubric", "consensus", "feedback", "protocol"]},

    {"id": "Q05", "text": "Do you evaluate the reliability and validity of the labels, especially when the true labels are inaccessible or difficult to define?",
     "category": "Missing features and labels", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe how you evaluate label reliability (e.g. Krippendorff's alpha, Cohen's kappa, test-retest).",
     "keywords": ["krippendorff", "cohen", "kappa", "alpha", "icc", "reliability",
                  "validity", "inter-annotator", "agreement", "test-retest", "cronbach", "consistency"]},

    {"id": "Q06", "text": "Does the dataset represent the diversity of the intended deployment environment (e.g., demographics, conditions, or scenarios)?",
     "category": "Non-representative sampling", "mode": "human", "q_type": "confirm", "risky_on": "No",
     "comment_prompt": "Optional: describe what demographic groups or scenarios are covered.", "keywords": []},

    {"id": "Q07", "text": "Do you implement techniques to measure and report the proportion of missing features or labels across sensitive groups?",
     "category": "Missing features and labels", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the technique (e.g. per-subgroup missingness rates, MCAR/MAR/MNAR tests, dashboards).",
     "keywords": ["missing rate", "missingness", "mcar", "mar", "mnar", "subgroup",
                  "report", "dashboard", "proportion", "completeness", "audit", "nan", "null"]},

    {"id": "Q08", "text": "Do you use sampling techniques like stratified sampling or oversampling to address underrepresented groups?",
     "category": "Non-representative sampling", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — name the technique (e.g. SMOTE, stratified sampling, oversampling, undersampling, class weighting).",
     "keywords": ["smote", "stratif", "oversamp", "undersamp", "resamp", "weight",
                  "balance", "bootstrap", "augment", "synthetic", "adasyn", "tomek", "near miss"]},

    {"id": "Q09", "text": "Do you evaluate potential over-representation of specific demographic groups or scenarios and assess its impact?",
     "category": "Non-representative sampling", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe how you detect over-representation (e.g. CI metric, frequency analysis, demographic breakdown).",
     "keywords": ["ci", "class imbalance", "frequency", "proportion", "demographic",
                  "breakdown", "distribution", "dominant", "majority", "over-represent", "ratio"]},

    {"id": "Q10", "text": "Do you use external benchmarking datasets to compare the representativeness of your sample?",
     "category": "Non-representative sampling", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — name the external dataset(s) or census/registry used for comparison.",
     "keywords": ["census", "benchmark", "reference", "external", "registry",
                  "national", "population", "database", "cohort", "comparison", "baseline", "published"]},

    {"id": "Q11", "text": "Do you use any methods to ensure fairness across different groups?",
     "category": "Selection bias", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the fairness method (e.g. re-weighting, adversarial debiasing, calibration, disparate impact removal).",
     "keywords": ["re-weight", "reweigh", "adversar", "debiasing", "calibrat",
                  "disparate impact", "equalis", "equaliz", "fairness constraint", "demographic parity",
                  "equalized odds", "post-process", "pre-process", "mitigation"]},

    {"id": "Q12", "text": "Do you assess how missing labels or features impact subgroup performance and model interpretability?",
     "category": "Missing features and labels", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the method (e.g. subgroup error analysis, SHAP per subgroup, ablation study).",
     "keywords": ["subgroup", "shap", "lime", "ablation", "error analysis",
                  "performance gap", "disparity", "interpretab", "explainab", "feature importance", "attribution"]},

    {"id": "Q13", "text": "Do you use imputation techniques that do not introduce additional biases into the data?",
     "category": "Data processing", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — name the imputation method (e.g. MICE, KNN imputation, mean/median, multiple imputation).",
     "keywords": ["mice", "knn", "k-nearest", "mean", "median", "mode", "hot deck",
                  "multiple imput", "regression imput", "missforest", "iterative", "em imputation"]},

    {"id": "Q14", "text": "Do you evaluate whether missing labels are randomly distributed or correlated with sensitive attributes?",
     "category": "Missing features and labels", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the test used (e.g. Little's MCAR test, logistic regression on missingness, chi-square).",
     "keywords": ["little", "mcar", "mar", "mnar", "chi-square", "chi2",
                  "logistic", "correlation", "missing at random", "test", "pattern", "detect"]},

    {"id": "Q15", "text": "Do you assess whether missing data disproportionately affects certain groups, and how do you address the potential impact on model predictions for those groups?",
     "category": "Missing features and labels", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe BOTH: (1) how you assess per-group missingness, AND (2) how you address the impact (e.g. stratified imputation, group-specific weights, missingness flags).",
     "keywords": ["subgroup", "stratif", "imput", "weight", "flag", "indicator",
                  "group-specific", "per-group", "demographic", "sensitive", "missingness rate",
                  "address", "mitigat", "correct"]},

    # ── Data processing ──────────────────────────────────────────────────────
    {"id": "Q16", "text": "Do you apply any strategies for quality assessment of the used dataset?",
     "category": "Data processing", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the strategy (e.g. data profiling, completeness checks, duplicate detection, schema validation).",
     "keywords": ["profil", "completeness", "duplicate", "schema", "validat",
                  "quality check", "data audit", "clean", "outlier", "anomaly", "report"]},

    {"id": "Q17", "text": "Do you evaluate preprocessing steps (e.g., imputation, scaling) for their potential to introduce or amplify biases, particularly for underrepresented groups?",
     "category": "Data processing", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe which steps are audited and how (e.g. pre/post fairness metric comparison, subgroup check after scaling).",
     "keywords": ["pre/post", "before and after", "comparison", "subgroup", "audit",
                  "scale", "normaliz", "standariz", "fairness check", "bias check", "metric", "evaluat"]},

    {"id": "Q18", "text": "Do you ensure that preprocessing techniques, such as normalization or scaling, do not introduce biases?",
     "category": "Data processing", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the safeguard (e.g. group-specific normalization, bias-aware scaling, holdout verification).",
     "keywords": ["group-specific", "per-group", "aware", "holdout", "verification",
                  "check", "fairness", "separate", "stratif", "normaliz", "robust scal"]},

    {"id": "Q19", "text": "Do you document the rationale for data transformations and their potential impact on fairness?",
     "category": "Data processing", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe how transformations are documented (e.g. data card, model card, pipeline logs, README).",
     "keywords": ["data card", "model card", "log", "readme", "document", "record",
                  "pipeline", "provenance", "lineage", "version", "changelog", "audit trail"]},

    # ── Simpson's paradox ────────────────────────────────────────────────────
    {"id": "Q20", "text": "Do you analyze whether trends observed in individual groups reverse when data from these groups are combined?",
     "category": "Simpson's paradox", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the analysis (e.g. stratified vs. aggregate comparison, Simpson's paradox test, segmented regression).",
     "keywords": ["stratif", "segment", "simpson", "subgroup", "aggregate",
                  "disaggregate", "reversal", "trend", "conditional", "group comparison"]},

    {"id": "Q21", "text": "Do you conduct separate analyses for relevant subgroups to ensure trends are not masked?",
     "category": "Simpson's paradox", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — list which subgroups are analyzed separately (e.g. by gender, age group, vendor, ethnicity).",
     "keywords": ["gender", "age", "vendor", "ethnicity", "race", "subgroup",
                  "segment", "stratif", "separate", "disaggregat", "cohort"]},

    {"id": "Q22", "text": "Do you document subgroup-specific insights and compare them to overall findings for consistency?",
     "category": "Data aggregation", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe how insights are documented and compared (e.g. fairness report, dashboard, model card).",
     "keywords": ["report", "dashboard", "model card", "document", "record",
                  "compare", "consistency", "discrepancy", "subgroup", "summary"]},

    {"id": "Q23", "text": "Do you test whether removing or aggregating certain data alters the overall trends disproportionately?",
     "category": "Data aggregation", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the sensitivity test (e.g. leave-one-group-out, ablation, jackknife).",
     "keywords": ["leave-one", "ablation", "jackknife", "sensitivity", "bootstrap",
                  "subset", "removal", "perturbation", "test", "impact"]},

    {"id": "Q24", "text": "Do you analyze whether correlations between variables change direction or magnitude when stratified by sensitive attributes (e.g., gender, age, ethnicity)?",
     "category": "Simpson's paradox", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe how stratified correlations are measured (e.g. per-group Pearson/Spearman, interaction terms).",
     "keywords": ["pearson", "spearman", "correlation", "interaction", "moderat",
                  "stratif", "per-group", "conditional", "subgroup", "direction"]},

    {"id": "Q25", "text": "Do you implement any measures to identify and mitigate biases, such as out-group homogeneity bias, that may arise during data aggregation?",
     "category": "Data aggregation", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the measure (e.g. within-group variance check, cluster analysis, diversity index).",
     "keywords": ["within-group", "variance", "cluster", "diversity", "heterogeneit",
                  "homogeneit", "out-group", "aggregat", "mitigat", "index"]},

    {"id": "Q26", "text": "Do you validate aggregated data against disaggregated group-level analyses to identify inconsistencies?",
     "category": "Data aggregation", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the validation process (e.g. reconciliation report, automated checks, manual review).",
     "keywords": ["reconcil", "disaggregat", "group-level", "validat", "check",
                  "inconsistency", "comparison", "audit", "review", "cross-check"]},

    {"id": "Q27", "text": "Do you evaluate whether the aggregation method disproportionately amplifies or masks specific group trends?",
     "category": "Data aggregation", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — name the evaluation approach (e.g. weighted aggregation, influence functions, subgroup metric comparison).",
     "keywords": ["weight", "influence", "subgroup metric", "amplif", "mask",
                  "disproportionat", "evaluat", "comparison", "analysis"]},

    {"id": "Q28", "text": "Do you include metadata that captures subgroup characteristics to preserve transparency in aggregated results?",
     "category": "Data aggregation", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe what metadata is recorded (e.g. group size, demographic tags, data source flags).",
     "keywords": ["metadata", "tag", "flag", "demographic", "group size", "source",
                  "attribute", "annotate", "provenance", "document"]},

    {"id": "Q29", "text": "Do you test multiple aggregation methods to compare their impact on fairness and accuracy?",
     "category": "Data aggregation", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — list which aggregation methods were compared (e.g. mean, median, weighted mean, micro vs. macro average).",
     "keywords": ["mean", "median", "weighted", "micro", "macro", "harmonic",
                  "geometric", "aggregat", "compar", "multiple", "method"]},

    # ── Distributed training ─────────────────────────────────────────────────
    {"id": "Q30", "text": "Do you address potential biases caused by non-participation of data sources due to network issues, computing limitations, or regulatory restrictions?",
     "category": "Distributed training", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the approach (e.g. participation logs, fallback imputation, regulatory documentation).",
     "keywords": ["particip", "log", "fallback", "imputation", "regulat", "document",
                  "dropout", "node", "federat", "non-response", "missing node"]},

    {"id": "Q31", "text": "Do you evaluate the impact of uneven participation of nodes in distributed training on the model's fairness?",
     "category": "Distributed training", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe how uneven participation is measured and which fairness metrics are used per node.",
     "keywords": ["node", "participation", "federat", "fairness metric", "per-node",
                  "uneven", "weight", "contribution", "evaluat", "monitor"]},

    {"id": "Q32", "text": "Do you assess the quality of data contributions from each participating node to identify skewed distributions?",
     "category": "Distributed training", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the quality check method (e.g. per-node DBDM metrics, distribution comparison, data profiling).",
     "keywords": ["per-node", "node quality", "distribution", "dbdm", "profil",
                  "check", "skew", "imbalance", "assess", "monitor"]},

    {"id": "Q33", "text": "Do you implement mechanisms to ensure fairness across nodes with varying data volumes and quality?",
     "category": "Distributed training", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — describe the mechanism (e.g. federated averaging, node weighting, quality-gated aggregation).",
     "keywords": ["federat", "averaging", "weight", "quality-gat", "aggregat",
                  "balance", "mechanism", "equaliz", "normaliz", "threshold"]},

    {"id": "Q34", "text": "Do you use algorithms or methods (e.g., federated averaging) that account for biases introduced by heterogeneous data distributions?",
     "category": "Distributed training", "mode": "human", "q_type": "describe", "risky_on": "No",
     "comment_prompt": "If Yes — name the specific algorithm (e.g. FedAvg, FedProx, q-FedAvg, SCAFFOLD).",
     "keywords": ["fedavg", "fedprox", "qfedavg", "scaffold", "federated",
                  "heterogeneous", "algorithm", "method", "distributed"]},

    # ── AUTO questions ───────────────────────────────────────────────────────
    {"id": "Q35", "text": "Sampling bias — Is sampling bias detected in the dataset?",
     "category": "Selection bias", "mode": "auto", "q_type": "auto", "risky_on": "Yes",
     "auto_metrics": ["Class Imbalance (CI)", "Binary Ratio (BR)", "Binary Difference (BD)"],
     "auto_logic": "any_risky",
     "auto_explanation": "Auto-answered from DBDM metrics: CI (±0.1), BR (0.8–1.25), BD (±0.1)."},

    {"id": "Q36", "text": "Non-normality bias — Are distributional anomalies detected between groups?",
     "category": "Selection bias", "mode": "auto", "q_type": "auto", "risky_on": "Yes",
     "auto_metrics": ["KS value", "Jensen-Shannon Divergence (JS)", "Total Variation Distance (TVD)"],
     "auto_logic": "any_risky",
     "auto_explanation": "Auto-answered from DBDM metrics: KS (±0.1), JS (±0.1), TVD (±0.1)."},

    {"id": "Q37", "text": "Confounding variables — Are confounding associations detected between the facet and outcome?",
     "category": "Selection bias", "mode": "auto", "q_type": "auto", "risky_on": "Yes",
     "auto_metrics": ["Pearson Correlation (CORR)", "Normalized Mutual Information (NMI)"],
     "auto_logic": "any_risky",
     "auto_explanation": "Auto-answered from DBDM metrics: CORR (±0.1), NMI (±0.1)."},

    # ── Other sources of data bias ───────────────────────────────────────────
    {"id": "Q38", "text": "Does the dataset cover the intended population and scenarios adequately?",
     "category": "Other sources of data bias", "mode": "human", "q_type": "confirm", "risky_on": "No",
     "comment_prompt": "Optional: describe the intended population and how coverage was verified.", "keywords": []},

    {"id": "Q39", "text": "Are there any groups systematically excluded or less represented in the dataset?",
     "category": "Other sources of data bias", "mode": "human", "q_type": "confirm", "risky_on": "Yes",
     "comment_prompt": "Optional: list which groups are excluded or under-represented.", "keywords": []},
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _metric_is_risky(metric_name: str, value) -> bool:
    if value is None:
        return False
    threshold = DBDM_THRESHOLDS.get(metric_name)
    if threshold is None:
        return False
    if isinstance(threshold, tuple):
        lo, hi = threshold
        return not (lo <= float(value) <= hi)
    return abs(float(value)) > threshold


# ══════════════════════════════════════════════════════════════════════════════
# AUTO ANSWER
# ══════════════════════════════════════════════════════════════════════════════

def auto_answer(question: dict, dbdm_metrics: dict) -> dict:
    risky_metrics = {}
    all_vals = {}
    for m in question["auto_metrics"]:
        val = dbdm_metrics.get(m)
        if val is not None:
            fval = round(float(val), 4)
            all_vals[m] = fval
            if _metric_is_risky(m, val):
                risky_metrics[m] = fval

    is_risky = len(risky_metrics) > 0
    answer = "Yes" if is_risky else "No"

    if risky_metrics:
        fired = "; ".join(
            f"{k} = {v} (outside {THRESHOLD_DISPLAY.get(k, '±0.1')})"
            for k, v in risky_metrics.items()
        )
        explanation = f"Bias detected — {fired}. {question['auto_explanation']}"
    else:
        all_str = "; ".join(
            f"{k} = {v} (within {THRESHOLD_DISPLAY.get(k, '±0.1')})"
            for k, v in all_vals.items()
        )
        explanation = f"All metrics within thresholds — {all_str}. {question['auto_explanation']}"

    return {"answer": answer, "is_risky": is_risky, "explanation": explanation,
            "values": all_vals, "risky_metrics": risky_metrics}


# ══════════════════════════════════════════════════════════════════════════════
# FULL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all(human_answers: dict, dbdm_metrics: dict) -> list:
    """
    Build the base result list.
    - Auto questions: fully scored from DBDM metrics.
    - Human questions: raw answer stored; Claude evaluation added later in app.py.
    """
    results = []
    for q in QUESTIONS:
        qid = q["id"]
        if q["mode"] == "auto":
            auto = auto_answer(q, dbdm_metrics)
            results.append({
                "id": qid, "text": q["text"], "category": q["category"],
                "mode": "auto", "q_type": "auto",
                "answer": auto["answer"], "user_answer": "",
                "is_risky": auto["is_risky"], "risk_reason": auto["explanation"],
                "metric_values": auto["values"],
                "risky_metrics": auto["risky_metrics"],
                # Claude fields — empty for auto questions
                "claude_suggestion": "",
                "claude_score": None,
                "claude_verdict": "",
                "claude_feedback": "",
                "claude_key_gap": "",
            })
        else:
            ans = human_answers.get(qid, "")
            results.append({
                "id": qid, "text": q["text"], "category": q["category"],
                "mode": "human", "q_type": q.get("q_type", "confirm"),
                "answer": ans, "user_answer": ans,
                # Placeholders — filled by Claude evaluation in app.py
                "is_risky": True if not ans.strip() else None,
                "risk_reason": "Pending Claude evaluation." if ans.strip() else "No answer provided.",
                "metric_values": {}, "risky_metrics": {},
                "claude_suggestion": "",
                "claude_score": None,
                "claude_verdict": "",
                "claude_feedback": "",
                "claude_key_gap": "",
            })
    return results
