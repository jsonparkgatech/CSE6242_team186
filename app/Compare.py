# ---- standard header for app/*.py ----
from __future__ import annotations

from pathlib import Path
import sys

# Project root = repo folder (parent of /app)
ROOT = Path(__file__).resolve().parents[1]

# Ensure absolute imports work no matter how Streamlit is launched
if (p := str(ROOT)) not in sys.path:
    sys.path.insert(0, p)
SRC = ROOT / "src"
if (sp := str(SRC)) not in sys.path:
    sys.path.insert(0, sp)

# Convenience paths for configs/data/models (use these instead of relative strings)
CONFIGS = ROOT / "configs"
DATA_DIR = ROOT / "DATA"
MODELS_DIR = ROOT / "models"

# Optional: load .env if present (safe if missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass
# ---- end header ----
# 

from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from app.utils import load_service_or_error, render_nav, safe_serving_value
from src.scoring_system import apply_box_cox_normalization, assign_grades

FRIENDLY_NAMES = {
    "energy_kcal": "Calories",
    "protein_g": "Protein",
    "carbs_g": "Carbs",
    "fiber_g": "Fiber",
    "sugar_g": "Sugar",
    "added_sugar_g": "Added Sugar",
    "sodium_mg": "Sodium",
    "sat_fat_g": "Saturated Fat",
}

PREFER_HIGH = {"protein_g", "fiber_g"}
FRIENDLY_TO_BASE = {friendly: base for base, friendly in FRIENDLY_NAMES.items()}


def _compare_bucket() -> set[int]:
    bucket = st.session_state.setdefault("compare_ids", set())
    if not isinstance(bucket, set):
        bucket = set(bucket)
        st.session_state["compare_ids"] = bucket
    return bucket


def _apply_normalization(dataset: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Apply normalization to dataset and add normalized scores and grades."""
    if score_col in dataset.columns:
        # Apply normalization and grading
        original_scores = dataset[score_col]
        normalized_scores = apply_box_cox_normalization(original_scores)
        new_grades = assign_grades(normalized_scores)
        
        # Update dataset with new scores and grades
        dataset = dataset.copy()
        dataset[f"{score_col}_normalized"] = normalized_scores
        dataset["grade"] = new_grades
        
        return dataset
    else:
        return dataset


def _selection_ui(dataset: pd.DataFrame, score_col: str) -> List[int]:
    bucket = _compare_bucket()
    st.subheader("Compare items")
    if bucket:
        current = dataset[dataset["fdc_id"].isin(bucket)]
        cols = st.columns(len(current))
        for col, (_, row) in zip(cols, current.iterrows()):
            with col:
                st.metric(row["description"], f"Grade {row.get('grade', '—')}", help=f"Score {round(row.get(f'{score_col}_normalized', row.get(score_col, '—')), 0)}")
                if st.button("Remove", key=f"remove_{row['fdc_id']}"):
                    bucket.discard(int(row["fdc_id"]))

    available = dataset[~dataset["fdc_id"].isin(bucket)]
    if len(bucket) < 3 and not available.empty:
        # Use normalized score for sorting if available, otherwise use original score
        sort_col = f"{score_col}_normalized" if f"{score_col}_normalized" in available.columns else score_col
        options = available.sort_values(sort_col, ascending=False).head(1000)
        label_map = {
            f"{row.description} ({int(row.fdc_id)})": int(row.fdc_id)
            for row in options.itertuples()
        }
        selection = st.selectbox("Add an item", options=["—"] + list(label_map), index=0)
        if selection != "—" and st.button("Add to compare", key="add_compare"):
            bucket.add(label_map[selection])

    return sorted(bucket)


def _build_table(dataset: pd.DataFrame, ids: List[int], suffix: str) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()
    selected = dataset.set_index("fdc_id").loc[ids]
    columns = [selected.loc[idx, "description"] for idx in ids]
    data = {}
    for base, pretty in FRIENDLY_NAMES.items():
        col_name = f"{base}_{suffix}"
        if col_name not in selected.columns:
            continue
        values = selected[col_name]
        if values.notna().sum() == 0:
            continue
        data[pretty] = [values.loc[idx] for idx in ids]
    table = pd.DataFrame(data).T
    table.columns = columns
    return table


def _style_table(table: pd.DataFrame, suffix: str) -> "pd.io.formats.style.Styler":
    basic = table.copy()

    def highlight(row: pd.Series) -> List[str]:
        base = FRIENDLY_TO_BASE.get(row.name)
        prefer_high = base in PREFER_HIGH if base else False
        
        # Handle NaN values properly
        numeric_row = pd.to_numeric(row, errors='coerce')
        mask = numeric_row.notna()
        
        if not mask.any():
            return ["" for _ in row]
        
        valid_values = numeric_row[mask]
        target = valid_values.max() if prefer_high else valid_values.min()
        
        styles = []
        for value, valid in zip(numeric_row, mask):
            if not valid:
                styles.append("")
            elif np.isclose(value, target) and not np.isnan(target):
                styles.append("background-color: #2E8B57; color: white; font-weight: 600;")
            else:
                styles.append("")
        return styles

    styler = (
        basic.style.apply(highlight, axis=1)
        .format(lambda v: "—" if pd.isna(v) else f"{v:.1f}")
        .set_caption(f"Nutrients {suffix.replace('per', ' per ')}")
    )
    return styler


def main() -> None:
    st.set_page_config(page_title="Nutrition Explorer", page_icon="🥗", layout="wide")
    st.title("Compare Foods")
    render_nav("Compare")
    service = load_service_or_error()
    dataset = service.sample_dataset  # Use fast sample dataset for Compare
    
    # Apply normalization to get new scores and grades
    dataset = _apply_normalization(dataset, service.score_column)
    
    ids = _selection_ui(dataset, service.score_column)

    if not ids:
        st.info("Add foods to compare up to three items.")
        return

    score_col = service.score_column
    # Use normalized score column if available, otherwise use original
    sort_col = f"{score_col}_normalized" if f"{score_col}_normalized" in dataset.columns else score_col
    cards = dataset[dataset["fdc_id"].isin(ids)].sort_values(sort_col, ascending=False)
    card_cols = st.columns(len(cards))
    for col, (_, row) in zip(card_cols, cards.iterrows()):
        with col:
            # Display normalized score with 1.333x multiplier
            score_to_display = row.get(f"{score_col}_normalized", row.get(score_col, '—'))
            if score_to_display != '—':
                score_to_display = round(score_to_display, 0)
            st.metric(row["description"], f"Score {score_to_display}")
            st.caption(
                "Grade {grade} | kcal {kcal}".format(
                    grade=row.get("grade", "—"),
                    kcal=safe_serving_value(row, "energy_kcal_perserving", unit="kcal", decimals=0),
                )
            )

    tab_serv, tab_100g = st.tabs(["Per serving", "Per 100g"])
    with tab_serv:
        table = _build_table(dataset, ids, "perserving")
        if table.empty:
            st.info("Per-serving nutrients unavailable for the selected items.")
        else:
            st.dataframe(_style_table(table, "perserving"), width="stretch")
    with tab_100g:
        table = _build_table(dataset, ids, "per100g")
        if table.empty:
            st.info("Per-100g nutrients unavailable for the selected items.")
        else:
            st.dataframe(_style_table(table, "per100g"), width="stretch")

    st.caption("Tip: go back to the search page to add more foods to your compare basket.")


if __name__ == "__main__":
    main()
