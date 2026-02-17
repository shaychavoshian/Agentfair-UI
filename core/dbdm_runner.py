# core/dbdm_runner.py
"""
Thin wrapper around tools/DBDM.py.
Calls dbdm() in non-interactive mode and returns the parsed metrics dict.
"""

import sys
import os
import json
import importlib.util

# ── Dynamically load DBDM from the tools/ folder ──────────────────────────────
def _load_dbdm():
    here = os.path.dirname(os.path.abspath(__file__))
    dbdm_path = os.path.join(here, "..", "tools", "DBDM.py")
    spec = importlib.util.spec_from_file_location("DBDM", dbdm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DBDM = _load_dbdm()


def run_dbdm(
    file_path: str,
    facet: str,
    outcome: str,
    subgroup_col: str = "",
    label_value: float = 1.0,
) -> dict:
    """
    Run DBDM non-interactively on a CSV file.

    Returns a flat dict of {metric_name: float_value} from the 'Overall'
    section of the JSON output, or raises RuntimeError on failure.
    """
    # DBDM saves a JSON file next to the CSV.
    # We call it, then read the JSON back.
    json_path = DBDM.dbdm(
        interactive=False,
        file_path=file_path,
        facet=facet,
        outcome=outcome,
        subgroup_col=subgroup_col,
        label_value=label_value,
        subgroup_analysis=0,       # no cluster analysis by default
    )

    if not json_path or not os.path.exists(json_path):
        raise RuntimeError(
            "DBDM did not produce a JSON output file. "
            "Check that the facet/outcome column names are correct."
        )

    with open(json_path, "r") as f:
        raw = json.load(f)

    # DBDM puts everything under "Overall"
    overall = raw.get("Overall", {})

    # Remove non-numeric sub-dict (Risk level per metric) before returning
    metrics = {
        k: v
        for k, v in overall.items()
        if isinstance(v, (int, float)) and k != "Risk level per metric"
    }

    return metrics, json_path
