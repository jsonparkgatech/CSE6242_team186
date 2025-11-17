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

import math
import time
from typing import Any, Tuple
import hashlib

import pandas as pd
import streamlit as st

from app.utils import (
    find_items,
    find_items_optimized,
    load_service_or_error,
    page_link_button,
    render_nav,
    safe_serving_value,
)

# Cache for search results to avoid re-processing
@st.cache_data(show_spinner=False)
def _get_cached_search(dataset_hash: str, query: str, filters: dict, max_rows: int = 500) -> pd.DataFrame:
    """Cached search results to avoid re-processing."""
    # This will be called from the main search logic
    pass


def _init_compare_bucket() -> set[int]:
    bucket = st.session_state.setdefault("compare_ids", set())
    if not isinstance(bucket, set):
        bucket = set(bucket)
        st.session_state["compare_ids"] = bucket
    return bucket


def _apply_filters_optimized(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply filters efficiently and return filter state for caching."""
    form_options = ["All"]
    if "form" in df.columns:
        form_options += sorted(x for x in df["form"].dropna().unique())
    group_options = ["All"]
    if "food_category" in df.columns:
        group_options += sorted(x for x in df["food_category"].dropna().unique())

    with st.sidebar:
        st.header("Filters")
        low_sodium = st.checkbox("Low sodium (≤140mg/100g)", key="filter_low_sodium")
        high_fiber = st.checkbox("High fiber (≥5g/100g)", key="filter_high_fiber")
        form_choice = st.selectbox("Form", options=form_options, key="filter_form")
        group_choice = st.selectbox("Food group", options=group_options, key="filter_group")

    # Store filter state for caching
    filter_state = {
        "low_sodium": low_sodium,
        "high_fiber": high_fiber,
        "form_choice": form_choice,
        "group_choice": group_choice
    }

    # Apply filters more efficiently
    if df.empty:
        return df, filter_state
        
    filtered_indices = pd.Series(True, index=df.index)
    
    if low_sodium and "sodium_mg_per100g" in df.columns:
        filtered_indices &= df["sodium_mg_per100g"].le(140)
    if high_fiber and "fiber_g_per100g" in df.columns:
        filtered_indices &= df["fiber_g_per100g"].ge(5)
    if form_choice != "All" and "form" in df.columns:
        filtered_indices &= df["form"] == form_choice
    if group_choice != "All" and "food_category" in df.columns:
        filtered_indices &= df["food_category"] == group_choice
    
    return df[filtered_indices], filter_state


def _render_rows(df: pd.DataFrame, service) -> None:
    bucket = _init_compare_bucket()
    score_col = service.score_column
    page_size = 25
    total = len(df)
    pages = max(1, math.ceil(total / page_size))
    page = 0
    if pages > 1:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=pages,
            value=1,
            step=1,
        ) - 1
    start = page * page_size
    end = start + page_size
    subset = df.iloc[start:end]

    for _, row in subset.iterrows():
        food_id = int(row["fdc_id"])
        cols = st.columns([4, 1.5, 1, 1, 1.3, 1.6, 1.5])
        name = row.get("description", "Unknown item")
        group = row.get("food_category", "—")
        grade = row.get("grade", "—")
        score_val = _format_entry(row.get(score_col))
        kcal = safe_serving_value(row, "energy_kcal_perserving", unit="kcal", decimals=0)
        sodium = safe_serving_value(row, "sodium_mg_perserving", unit="mg", decimals=0)
        sugar = safe_serving_value(row, "sugar_g_perserving", unit="g", decimals=1)
        cols[0].markdown(f"**{name}**  \n`FDC {food_id}`")
        cols[1].markdown(f"Group\n`{group}`")
        cols[2].markdown(f"Grade\n`{grade}`")
        cols[3].markdown(f"Score\n`{score_val}`")
        cols[4].markdown(f"kcal/serv\n`{kcal}`")
        cols[5].markdown(f"Sodium/serv\n`{sodium}`  \nSugar/serv\n`{sugar}`")

        with cols[6]:
            page_link_button("Detail.py", label="Detail →", params={"fdc_id": food_id})
            disabled = len(bucket) >= 3 and food_id not in bucket
            if st.button("Compare +", key=f"compare_add_{food_id}", disabled=disabled):
                bucket.add(food_id)
        st.divider()


def _format_entry(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "—"


def _create_search_cache_key(dataset_hash: str, query: str, filters: dict) -> str:
    """Create a unique cache key for search results."""
    filter_str = str(sorted(filters.items()))
    cache_str = f"{dataset_hash}_{hash(query)}_{hash(filter_str)}"
    return hashlib.md5(cache_str.encode()).hexdigest()

def main() -> None:
    st.set_page_config(page_title="Nutrition Explorer", page_icon="🥗", layout="wide")
    st.title("Search Foods")
    render_nav("Search")
    service = load_service_or_error()
    dataset = service.dataset

    st.markdown(
        "Filter foods by name, brand, form, or nutrient shortcuts. "
        "Add interesting items to your compare basket or jump straight to the detail view."
    )

    # Debounced search input
    query = st.text_input("Search foods or brands…", value="", max_chars=60, key="search_input")
    
    # Get dataset hash for caching
    dataset_hash = hashlib.md5(str(dataset.shape).encode()).hexdigest()
    
    # Initialize session state for debouncing
    if "last_search_time" not in st.session_state:
        st.session_state.last_search_time = 0
    if "debounced_query" not in st.session_state:
        st.session_state.debounced_query = ""

    # Debounce search (wait 0.5 seconds after user stops typing)
    current_time = time.time()
    if query != st.session_state.debounced_query:
        if current_time - st.session_state.last_search_time > 0.5:
            st.session_state.debounced_query = query
            st.session_state.last_search_time = current_time

    debounced_query = st.session_state.debounced_query

    # Show search status
    if query and query != debounced_query:
        st.info("🔍 Searching...")
    elif debounced_query:
        st.success(f"🔍 Searching for: {debounced_query}")

    # Perform search with optimized function
    with st.spinner("Processing search..."):
        if len(debounced_query) >= 2:
            # Use optimized search
            base = find_items_optimized(dataset, debounced_query, max_rows=500)
        else:
            # Show top scored items (no search needed)
            if len(dataset) > 1000:
                base = dataset.nlargest(200, service.score_column)
            else:
                base = dataset.nlargest(min(200, len(dataset)), service.score_column)

    # Apply filters with optimization
    filtered, filter_state = _apply_filters_optimized(base)
    
    # Sort results
    if not filtered.empty:
        filtered = filtered.sort_values(service.score_column, ascending=False)
        # Limit to 500 results for performance
        filtered = filtered.head(500)

    # Show results
    if filtered.empty:
        message = (
            "Type at least two characters to search." if len(debounced_query) < 2 else "No foods matched the filters."
        )
        st.info(message)
        return

    # Results summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Results", len(filtered))
    with col2:
        st.metric("Avg Score", f"{filtered[service.score_column].mean():.1f}" if service.score_column in filtered.columns else "N/A")
    with col3:
        if "grade" in filtered.columns:
            grade_a_b = (filtered["grade"].isin(["A", "B"]).mean() * 100) if "grade" in filtered.columns else 0
            st.metric("Grade A/B %", f"{grade_a_b:.1f}%")

    # Download button
    csv_bytes = filtered[
        ["fdc_id", "description", "food_category", service.score_column, "grade"]
    ].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="search_results.csv",
        mime="text/csv",
    )

    # Progressive loading for large result sets
    if len(filtered) > 100:
        st.info(f"📊 Showing first 100 of {len(filtered)} results. Loading more...")
        filtered = filtered.head(100)

    _render_rows(filtered, service)


if __name__ == "__main__":
    main()
