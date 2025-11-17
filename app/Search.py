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
from src.scoring_system import apply_box_cox_normalization, assign_grades, get_score_statistics

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
    
    # Grade options
    grade_options = ["All"]
    if "grade" in df.columns:
        grade_options += sorted(x for x in df["grade"].dropna().unique())
    
    # Brand options (top brands for better selection)
    brand_options = ["All"]
    if "brand_owner" in df.columns:
        top_brands = df["brand_owner"].value_counts().head(100).index.tolist()
        brand_options += [brand for brand in top_brands if brand and brand != ""]

    with st.sidebar:
        st.header("🔍 Advanced Filters")
        
        # Basic filters - all visible and expanded
        st.subheader("✓ Basic Nutrient Filters")
        low_sodium = st.checkbox("🥬 Low sodium (≤140mg/100g)", key="filter_low_sodium")
        high_fiber = st.checkbox("🌾 High fiber (≥5g/100g)", key="filter_high_fiber")
        low_sugar = st.checkbox("🍯 Low sugar (≤5g/100g)", key="filter_low_sugar")
        high_protein = st.checkbox("🥩 High protein (≥10g/100g)", key="filter_high_protein")
        
        st.markdown("---")  # Visual separator
        
        # Grade filter - expanded with clear options
        st.subheader("📊 Grade Filter")
        grade_choice = st.selectbox(
            "Select Grade", 
            options=grade_options, 
            key="filter_grade",
            help="Filter foods by nutrition grade"
        )
        
        st.markdown("---")  # Visual separator
        
        # Food characteristics - all visible
        st.subheader("🏷️ Food Characteristics")
        form_choice = st.selectbox(
            "Food Form", 
            options=form_options, 
            key="filter_form",
            help="Filter by solid, liquid, or other forms"
        )
        group_choice = st.selectbox(
            "Food Group", 
            options=group_options, 
            key="filter_group",
            help="Filter by USDA food category"
        )
        brand_choice = st.selectbox(
            "Brand", 
            options=brand_options, 
            key="filter_brand",
            help="Filter by manufacturer/brand name"
        )
        
        st.markdown("---")  # Visual separator
        
        # Nutrient ranges - all sliders visible
        st.subheader("📈 Nutrient Ranges (per 100g)")
        
        # Initialize range variables with wider default ranges
        kcal_range = (0, 500)
        protein_range = (0, 50)
        carbs_range = (0, 100)
        fat_range = (0, 50)
        sodium_range = (0, 2000)
        score_range = (0, 100)
        
        # Calorie range
        if "energy_kcal_per100g" in df.columns:
            kcal_data = df["energy_kcal_per100g"].dropna()
            if len(kcal_data) > 0:
                kcal_min, kcal_max = int(kcal_data.min()), int(kcal_data.max())
                kcal_range = st.slider(
                    "🔥 Calories (kcal)", 
                    min_value=0, 
                    max_value=max(500, kcal_max), 
                    value=(0, max(300, kcal_max)), 
                    key="filter_kcal_range",
                    help="Set calorie range per 100g"
                )
        
        # Protein range
        if "protein_g_per100g" in df.columns:
            protein_data = df["protein_g_per100g"].dropna()
            if len(protein_data) > 0:
                protein_min, protein_max = int(protein_data.min()), int(protein_data.max())
                protein_range = st.slider(
                    "🥩 Protein (g)", 
                    min_value=0, 
                    max_value=max(50, protein_max), 
                    value=(0, max(30, protein_max)), 
                    key="filter_protein_range",
                    help="Set protein range per 100g"
                )
        
        # Carbs range
        if "carbs_g_per100g" in df.columns:
            carbs_data = df["carbs_g_per100g"].dropna()
            if len(carbs_data) > 0:
                carbs_min, carbs_max = int(carbs_data.min()), int(carbs_data.max())
                carbs_range = st.slider(
                    "🍞 Carbs (g)", 
                    min_value=0, 
                    max_value=max(100, carbs_max), 
                    value=(0, max(50, carbs_max)), 
                    key="filter_carbs_range",
                    help="Set carbohydrate range per 100g"
                )
        
        # Fat range
        if "total_fat_g_per100g" in df.columns:
            fat_data = df["total_fat_g_per100g"].dropna()
            if len(fat_data) > 0:
                fat_min, fat_max = int(fat_data.min()), int(fat_data.max())
                fat_range = st.slider(
                    "🥑 Total Fat (g)", 
                    min_value=0, 
                    max_value=max(50, fat_max), 
                    value=(0, max(25, fat_max)), 
                    key="filter_fat_range",
                    help="Set total fat range per 100g"
                )
        
        # Sodium range
        if "sodium_mg_per100g" in df.columns:
            sodium_data = df["sodium_mg_per100g"].dropna()
            if len(sodium_data) > 0:
                sodium_min, sodium_max = int(sodium_data.min()), int(sodium_data.max())
                sodium_range = st.slider(
                    "🧂 Sodium (mg)", 
                    min_value=0, 
                    max_value=max(2000, sodium_max), 
                    value=(0, max(1000, sodium_max)), 
                    key="filter_sodium_range",
                    help="Set sodium range per 100g"
                )
        
        # Score range
        score_col = "score_normalized" if "score_normalized" in df.columns else "score"
        if score_col in df.columns:
            score_data = df[score_col].dropna()
            if len(score_data) > 0:
                score_min, score_max = int(score_data.min()), int(score_data.max())
                score_range = st.slider(
                    "⭐ Nutrition Score", 
                    min_value=0, 
                    max_value=100, 
                    value=(0, 100), 
                    key="filter_score_range",
                    help="Set nutrition score range (higher is better)"
                )

        st.markdown("---")  # Visual separator
        st.caption("💡 Tip: Combine multiple filters for precise results")

    # Store filter state for caching
    filter_state = {
        "low_sodium": low_sodium,
        "high_fiber": high_fiber,
        "low_sugar": low_sugar,
        "high_protein": high_protein,
        "grade_choice": grade_choice,
        "form_choice": form_choice,
        "group_choice": group_choice,
        "brand_choice": brand_choice,
        "kcal_range": kcal_range,
        "protein_range": protein_range,
        "carbs_range": carbs_range,
        "fat_range": fat_range,
        "sodium_range": sodium_range,
        "score_range": score_range,
    }

    # Apply filters more efficiently
    if df.empty:
        return df, filter_state
        
    filtered_indices = pd.Series(True, index=df.index)
    
    # Basic nutrient filters
    if low_sodium and "sodium_mg_per100g" in df.columns:
        filtered_indices &= df["sodium_mg_per100g"].le(140)
    if high_fiber and "fiber_g_per100g" in df.columns:
        filtered_indices &= df["fiber_g_per100g"].ge(5)
    if low_sugar and "sugar_g_per100g" in df.columns:
        filtered_indices &= df["sugar_g_per100g"].le(5)
    if high_protein and "protein_g_per100g" in df.columns:
        filtered_indices &= df["protein_g_per100g"].ge(10)
    
    # Category filters
    if grade_choice != "All" and "grade" in df.columns:
        filtered_indices &= df["grade"] == grade_choice
    if form_choice != "All" and "form" in df.columns:
        filtered_indices &= df["form"] == form_choice
    if group_choice != "All" and "food_category" in df.columns:
        filtered_indices &= df["food_category"] == group_choice
    if brand_choice != "All" and "brand_owner" in df.columns:
        filtered_indices &= df["brand_owner"] == brand_choice
    
    # Range filters
    if "energy_kcal_per100g" in df.columns:
        kcal_min, kcal_max = kcal_range
        filtered_indices &= df["energy_kcal_per100g"].between(kcal_min, kcal_max)
    
    if "protein_g_per100g" in df.columns:
        protein_min, protein_max = protein_range
        filtered_indices &= df["protein_g_per100g"].between(protein_min, protein_max)
    
    if "carbs_g_per100g" in df.columns:
        carbs_min, carbs_max = carbs_range
        filtered_indices &= df["carbs_g_per100g"].between(carbs_min, carbs_max)
    
    if "total_fat_g_per100g" in df.columns:
        fat_min, fat_max = fat_range
        filtered_indices &= df["total_fat_g_per100g"].between(fat_min, fat_max)
    
    if "sodium_mg_per100g" in df.columns:
        sodium_min, sodium_max = sodium_range
        filtered_indices &= df["sodium_mg_per100g"].between(sodium_min, sodium_max)
    
    if score_col in df.columns:
        score_min, score_max = score_range
        filtered_indices &= df[score_col].between(score_min, score_max)
    
    return df[filtered_indices], filter_state


def _render_rows(df: pd.DataFrame, service) -> None:
    bucket = _init_compare_bucket()
    score_col = service.score_column
    page_size = 100  # Increased page size for better performance
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
        
        # Use normalized score if available, otherwise original score (matching Home page logic)
        normalized_score_col = f"{score_col}_normalized"
        if normalized_score_col in row.index and pd.notna(row.get(normalized_score_col)):
            score_val = f"{round(float(row.get(normalized_score_col, 0)), 0)}"
        else:
            score_val = f"{round(float(row.get(score_col, 0)), 0)}"
            
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
    # Use FULL dataset instead of sample dataset to get all 50k+ items
    dataset = service.dataset

    # Apply new scoring system to search dataset (same logic as Home page)
    score_col = service.score_column
    normalized_scores = pd.Series([], dtype=float)
    if score_col in dataset.columns:
        original_scores = dataset[score_col]
        normalized_scores = apply_box_cox_normalization(original_scores)
        new_grades = assign_grades(normalized_scores)
        
        dataset = dataset.copy()
        dataset[f"{score_col}_normalized"] = normalized_scores
        dataset["grade"] = new_grades

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

    # Perform search with optimized function - REMOVE ALL LIMITS
    with st.spinner("Processing search..."):
        if len(debounced_query) >= 2:
            # Use optimized search - NO MAX ROWS LIMIT
            base = find_items_optimized(dataset, debounced_query, max_rows=50000)  # Increased to 50k
        else:
            # Show all items when no search query (NO LIMIT)
            sort_col = f"{score_col}_normalized" if f"{score_col}_normalized" in dataset.columns else score_col
            base = dataset.sort_values(sort_col, ascending=False)  # Sort all items, no limit

    # Apply filters with optimization
    filtered, filter_state = _apply_filters_optimized(base)
    
    # Sort results
    if not filtered.empty:
        # Sort by normalized score if available, otherwise original score
        sort_col = f"{score_col}_normalized" if f"{score_col}_normalized" in filtered.columns else score_col
        filtered = filtered.sort_values(sort_col, ascending=False)
        # NO LIMITS - show all results

    # Show results
    if filtered.empty:
        message = (
            "Type at least two characters to search." if len(debounced_query) < 2 else "No foods matched the filters."
        )
        st.info(message)
        return

    # Results summary (using EXACT same calculation as Home page for consistency)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Results", len(filtered))
    with col2:
        # Calculate statistics using EXACT same logic as Home page (global dataset stats)
        if len(normalized_scores) > 0:
            score_stats = get_score_statistics(normalized_scores)
            avg_score = round(score_stats['mean'], 0)
            st.metric("Avg Score (normalized)", f"{avg_score}")
        else:
            st.metric("Avg Score (normalized)", "N/A")
    with col3:
        if "grade" in filtered.columns:
            grade_a_b = (filtered["grade"].isin(["A", "B"]).mean() * 100) if "grade" in filtered.columns else 0
            st.metric("Grade A/B %", f"{grade_a_b:.1f}%")

    # Download button
    display_cols = ["fdc_id", "description", "food_category"]
    if f"{service.score_column}_normalized" in filtered.columns:
        display_cols.extend([f"{service.score_column}_normalized", "grade"])
    else:
        display_cols.extend([service.score_column, "grade"])
    
    csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="search_results.csv",
        mime="text/csv",
    )

    # Show total dataset info
    st.info(f"📊 Total dataset size: {len(dataset):,} foods | Showing: {len(filtered):,} results")

    _render_rows(filtered, service)


if __name__ == "__main__":
    main()
