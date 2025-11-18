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
import plotly.express as px
import plotly.graph_objects as go

from app.utils import load_service_or_error, render_nav, quick_nav_button
from src.scoring_system import apply_box_cox_normalization, assign_grades, get_score_statistics, get_grade_distribution

def main() -> None:
    st.set_page_config(page_title="Nutrition Explorer", page_icon=":material/restaurant:", layout="wide")
    st.title("Nutrition Explorer")
    render_nav("Home")
    service = load_service_or_error()
    dataset = service.dataset
    score_col = service.score_column

    # Project description
    st.markdown("""
    ## About Nutrition Explorer
    
    **Nutrition Explorer** addresses the growing need for accessible, data-driven nutrition guidance in an era where consumers are increasingly health-conscious but lack reliable tools to make informed food choices. Our platform empowers users to discover healthier food alternatives through comprehensive nutrient analysis, personalized scoring systems, and intelligent substitution recommendations. Key features include advanced food search with multi-dimensional filtering, detailed nutritional breakdowns, side-by-side food comparisons, and AI-powered substitute suggestions that consider both nutritional improvements and user preferences. The application serves consumers, nutritionists, dietitians, and food product developers seeking to understand food quality, make better dietary decisions, and identify healthier alternatives that maintain taste preferences while optimizing nutritional profiles.
    
    Built with modern data science and machine learning technologies, Nutrition Explorer processes over 350,000 food items from the USDA FoodData Central database using advanced normalization techniques and sophisticated scoring algorithms. The platform employs Facebook's embedding technology for fast similarity searches, implements Box-Cox normalization for score distribution optimization, and uses a grade-based classification system (A-F) to make complex nutritional data accessible to non-experts. The technical architecture prioritizes performance through feature reduction, optimized algorithms, and efficient data processing pipelines, ensuring real-time responses even with large-scale nutritional datasets.
    """)

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

    # Grade distribution pie chart using new grading system with 1.333x multiplier
    st.subheader("Grade distribution (A: 80-100, B: 60-80, C: 40-60, D: 20-40, F: 0-20)")
    if grade_dist:
        total_grades = sum(grade_dist.values())
        grade_pcts = {grade: round(count / total_grades * 100, 1) for grade, count in grade_dist.items()}
        
        # Create pie chart with better colors
        grades = list(grade_pcts.keys())
        percentages = list(grade_pcts.values())
        colors = ['#2E8B57', '#90EE90', '#FFD700', '#FFA500', '#FF6B6B']  # Green to red
        
        fig = px.pie(
            values=percentages,
            names=grades,
            color_discrete_sequence=colors,
            title="Distribution of Nutrition Grades"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Based on normalized 1-100 scoring system with 1.333x multiplier")
    else:
        st.info("No grade distribution available.")

    st.caption(
        "Data sourced from USDA FoodData Central. Scores and grades use the new normalized 1-100 system with 1.333x multiplier."
    )


if __name__ == "__main__":
    main()
