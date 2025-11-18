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

import pandas as pd
import streamlit as st
import numpy as np

from app.utils import load_service_or_error, render_nav, quick_nav_button
from src.scoring_system import apply_box_cox_normalization, assign_grades, get_score_statistics, get_grade_distribution

def main() -> None:
    st.set_page_config(page_title="Nutrition Explorer", page_icon=":material/restaurant:", layout="wide")
    st.title("Nutrition Explorer")
    render_nav("Home")
    service = load_service_or_error()
    dataset = service.dataset
    score_col = service.score_column

    st.markdown(
        """
        Explore USDA FoodData Central foods, track nutrient strengths, and discover
        healthier swaps powered by our scoring model. Use the quick actions below to dive in.
        
        Visualization uses a random subset.
        """
    )

    # Use the new normalized scoring system with 1.333x multiplier
    if score_col in dataset.columns:
        # Apply normalization and grading
        original_scores = dataset[score_col]
        normalized_scores = apply_box_cox_normalization(original_scores)
        new_grades = assign_grades(normalized_scores)
        
        # Update dataset with new scores and grades
        dataset = dataset.copy()
        dataset[f"{score_col}_normalized"] = normalized_scores
        dataset["grade"] = new_grades
        
        # Calculate statistics from normalized scores
        stats = get_score_statistics(normalized_scores)
        grade_dist = get_grade_distribution(new_grades)
    else:
        # Fallback if score column not found
        stats = {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "count": 0}
        grade_dist = {}

    metrics = st.columns(4)
    with metrics[0]:
        st.metric("Food in total dataset", f"{len(dataset):,}")
    with metrics[1]:
        st.metric("Average score (normalized)", f"{round(stats['mean'], 0)}")
    with metrics[2]:
        st.metric("Median score (normalized)", f"{round(stats['median'], 0)}")
    with metrics[3]:
        st.metric("Std deviation", f"{round(stats['std'], 0)}")

    # Score distribution histogram (using normalized scores with 1.333x multiplier)
    if len(normalized_scores) > 0:
        st.subheader("Score distribution (normalized 1-100)")
        
        # Create histogram bins for normalized scores
        min_score = normalized_scores.min()
        max_score = normalized_scores.max()
        
        # Create 20 bins
        bins = pd.cut(normalized_scores, bins=20, precision=0)
        bin_counts = bins.value_counts().sort_index()
        
        # Format bin labels
        bin_labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in bin_counts.index]
        
        # Create DataFrame for chart
        hist_data = pd.DataFrame({'Count': bin_counts.values}, index=bin_labels)
        
        # Display histogram
        st.bar_chart(hist_data)
        st.caption(f"Normalized score range: {min_score:.1f} to {max_score:.1f} | Foods with scores: {stats['count']:,}")
    else:
        st.subheader("Score distribution")
        st.info("No valid scores found for distribution analysis.")

    # Feature histograms
    st.subheader("Feature distributions")
    
    # Define nutrient features (removed protein and carbs)
    nutrient_features = [
        "total_fat_g_per100g",
        "energy_kcal_per100g",
        "sugar_g_per100g"
    ]
    
    # Create columns for feature histograms
    feature_cols = st.columns(len(nutrient_features))
    
    for i, feature in enumerate(nutrient_features):
        with feature_cols[i]:
            if feature in dataset.columns:
                feature_data = dataset[feature].dropna()
                if len(feature_data) > 0:
                    # Create histogram for this feature
                    min_val = feature_data.min()
                    max_val = feature_data.max()
                    
                    # Sample if too many values
                    if len(feature_data) > 1000:
                        feature_sample = feature_data.sample(1000)
                    else:
                        feature_sample = feature_data
                    
                    # Create bins
                    bins = pd.cut(feature_sample, bins=10, precision=0)
                    bin_counts = bins.value_counts().sort_index()
                    
                    if len(bin_counts) > 0:
                        chart_data = pd.DataFrame({'Count': bin_counts.values})
                        st.bar_chart(chart_data)
                        st.caption(f"{feature.replace('_', ' ').title()}")
            else:
                st.caption(f"{feature.replace('_', ' ').title()}: Not available")

    # Grade distribution chart using new grading system with 1.333x multiplier
    st.subheader("Grade distribution (A: 80-100, B: 60-80, C: 40-60, D: 20-40, F: 0-20)")
    if grade_dist:
        total_grades = sum(grade_dist.values())
        grade_pcts = {grade: round(count / total_grades * 100, 0) for grade, count in grade_dist.items()}
        
        # Create horizontal bar chart (grades on y-axis, percentage on x-axis)
        chart_data = pd.DataFrame({
            'Percentage': list(grade_pcts.values())
        }, index=list(grade_pcts.keys()))
        
        st.bar_chart(chart_data)
        st.caption("Based on normalized 1-100 scoring system with 1.333x multiplier")
    else:
        st.info("No grade distribution available.")

    st.divider()
    st.subheader("Nutrient highlights")
    
    # Define nutrient features for statistics
    nutrient_features_stats = [
        "protein_g_per100g",
        "total_fat_g_per100g", 
        "carbs_g_per100g",
        "energy_kcal_per100g",
        "sugar_g_per100g",
        "sodium_mg_per100g",
        "fiber_g_per100g"
    ]
    
    # Calculate and display statistics for each feature
    stats_data = []
    for feature in nutrient_features_stats:
        if feature in dataset.columns:
            feature_data = dataset[feature].dropna()
            if len(feature_data) > 0:
                stats_data.append({
                    'Nutrient': feature.replace('_', ' ').title(),
                    'Mean': round(feature_data.mean(), 0),
                    'Median': round(feature_data.median(), 0),
                    'Std Dev': round(feature_data.std(), 0)
                })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, width='stretch')
    else:
        st.info("No nutrient data available for statistics.")

    st.caption(
        "Data sourced from USDA FoodData Central. Scores and grades use the new normalized 1-100 system with 1.333x multiplier."
    )


if __name__ == "__main__":
    main()
