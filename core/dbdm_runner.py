# core/dbdm_runner.py
"""
Thin wrapper around tools/DBDM.py.
Calls dbdm() in non-interactive mode and returns parsed metrics + risk levels.

Fix 1: Normalises the facet column to 0/1 before passing to DBDM.
        DBDM internally looks for facet values 0 and 1 by index position
        (iloc[0], iloc[1]). If the facet contains e.g. 1 and 2, DPL/BR/BD/LR
        all fail with ZeroDivisionError and calculate_metrics() returns None,
        leaving the JSON Overall dict empty.

Fix 2: Returns risk_levels dict alongside metrics so the UI can display
        DBDM's own severity labels (No risk / Low / Medium / High / Very high).
"""

import sys
import os
import json
import tempfile
import importlib.util

import pandas as pd


# ── Dynamically load DBDM from the tools/ folder ──────────────────────────────
def _load_dbdm():
    here      = os.path.dirname(os.path.abspath(__file__))
    dbdm_path = os.path.join(here, "..", "tools", "DBDM.py")
    spec      = importlib.util.spec_from_file_location("DBDM", dbdm_path)
    mod       = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DBDM = _load_dbdm()


def _normalise_facet(df: pd.DataFrame, facet: str) -> pd.DataFrame:
    """
    Remap the facet column to 0/1 if it isn't already.
    DBDM's binary metrics (DPL, BR, BD, LR) require exactly the values 0 and 1.
    """
    unique_vals = sorted(df[facet].dropna().unique())
    if list(unique_vals) == [0, 1]:
        return df                       # already correct — nothing to do
    if len(unique_vals) != 2:
        return df                       # non-binary facet — let DBDM handle/skip
    mapping     = {unique_vals[0]: 0, unique_vals[1]: 1}
    df          = df.copy()
    df[facet]   = df[facet].map(mapping)
    return df


def run_dbdm(
    file_path: str,
    facet: str,
    outcome: str,
    subgroup_col: str = "",
    label_value: float = 1.0,
) -> tuple:
    """
    Run DBDM non-interactively on a CSV file.

    Returns
    -------
    metrics      : dict  {metric_name: float}
    risk_levels  : dict  {metric_name: "No risk"|"Low risk"|...|"Very high risk"}
    json_path    : str   path to the raw DBDM JSON output file
    """
    # ── Load, normalise, save to a fresh temp file ────────────────────────────
    df          = pd.read_csv(file_path)
    df          = _normalise_facet(df, facet)

    tmp         = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    normalised_path = tmp.name

    # ── Run DBDM ──────────────────────────────────────────────────────────────
    json_path = DBDM.dbdm(
        interactive      = False,
        file_path        = normalised_path,
        facet            = facet,
        outcome          = outcome,
        subgroup_col     = subgroup_col,
        label_value      = label_value,
        subgroup_analysis= 0,
    )

    if not json_path or not os.path.exists(json_path):
        raise RuntimeError(
            "DBDM did not produce a JSON output file. "
            "Check that the facet/outcome column names are correct."
        )

    # ── Parse JSON ────────────────────────────────────────────────────────────
    with open(json_path, "r") as f:
        raw = json.load(f)

    overall      = raw.get("Overall", {})
    risk_levels  = overall.get("Risk level per metric", {})

    metrics = {
        k: v
        for k, v in overall.items()
        if isinstance(v, (int, float))
    }

    if not metrics:
        raise RuntimeError(
            "DBDM ran but returned no metrics. "
            "Ensure the facet and outcome columns exist and contain valid data."
        )

    return metrics, risk_levels, json_path
