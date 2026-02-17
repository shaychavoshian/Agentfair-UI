# AgentFair — Interactive AI Bias Detection UI

> ISO 24027 Enhanced Questionnaire + DBDM Toolkit  
> Streamlit-based interactive bias detection for AI datasets

---

## What it does

AgentFair walks you through the complete ISO 24027 data bias assessment:

1. **Upload** your CSV dataset
2. **DBDM metrics** are calculated automatically (15 fairness metrics)
3. **3 questions auto-answered** from DBDM results (Q35, Q36, Q37)
4. **36 questions asked interactively** one by one (Yes / No)
5. **Full report** shows each question as Risky ⚠️ or Not Risky ✅ — no score, just the flag

---

## Project Structure

```
agentfair_ui/
├── app.py                    ← Streamlit entry point (run this)
├── requirements.txt
├── .gitignore
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── questionnaire.py      ← All 39 ISO questions + auto-answer logic
│   └── dbdm_runner.py        ← Thin wrapper around DBDM.py
│
├── tools/
│   └── DBDM.py               ← Original DBDM toolkit (unchanged)
│
└── sample_data/
    └── example.csv           ← Small test dataset (vendor / label / age_group)
```

---

## Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/agentfair-ui.git
cd agentfair-ui
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## How to Use

| Step | What happens |
|------|-------------|
| **1 — Upload** | Upload any CSV. Select the **facet** (sensitive attribute), **outcome** (label), and optionally a **subgroup** column. |
| **2 — DBDM** | 15 bias metrics are computed. Q35/Q36/Q37 are auto-answered. You see a table with ✅ Fair / ⚠️ Risky per metric. |
| **3 — Questionnaire** | 36 questions asked one by one. Click Yes or No. You can go back and re-answer. Auto-answered questions are shown in a collapsible panel. |
| **4 — Report** | Every question listed with ⚠️ RISKY or ✅ Not Risky. Download full JSON report or CSV. |

---

## The 3 Auto-answered Questions (from DBDM)

| Question | DBDM Metrics Used | Risky if |
|----------|------------------|---------|
| **Q35** — Sampling bias | CI, BR, BD | Any metric outside threshold |
| **Q36** — Non-normality | KS, JS, TVD | Any metric > 0.1 |
| **Q37** — Confounding variables | CORR, NMI | Any metric > 0.1 |

---

## Fairness Thresholds (from DBDM.py)

| Metric | Threshold |
|--------|-----------|
| CI, DPL, DD, JS, KS, NMI, BD, CORR, TVD, CDD, NCMI, CBD, LR | ±0.1 |
| BR (Binary Ratio) | Must be in range 0.8 – 1.25 |
| LR Intercept | Must be in range −2 to +2 |

---

## Publishing to GitHub

```bash
# Inside the project folder:
git init
git add .
git commit -m "Initial commit: AgentFair bias detection UI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/agentfair-ui.git
git push -u origin main
```

---

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`
- No API keys needed (fully local)

---

## Based on

- DBDM Toolkit (FAITH project)
- ISO/IEC 24027 — *Bias in AI systems and AI aided decision making*
- Book chapter: *"Mitigating bias and enhancing trust in AI-based systems through a synthetic data-driven framework"*
