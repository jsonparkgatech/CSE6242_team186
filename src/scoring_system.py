"""Scoring and grading utilities with normalized 1-100 system."""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "DATA" / "processed"

# Grade boundaries for 1-100 normalized scoring system with 1.333x multiplier
GRADE_BOUNDARIES = {
    "A": 80,
    "B": 60, 
    "C": 40,
    "D": 20,
    "F": 0
}

def apply_box_cox_normalization(scores: pd.Series) -> pd.Series:
    """
    Apply Box-Cox-like transformation and normalize to 1-100 scale with 1.333x multiplier.
    """
    # Filter out scores of 0 and NaN
    valid_scores = scores[(scores > 0) & scores.notna()].copy()
    
    if len(valid_scores) == 0:
        return pd.Series([], dtype=float)
    
    # Add small constant to avoid log(0) and make all values positive
    shifted_scores = valid_scores + 1
    
    # Apply log transformation (Box-Cox-like)
    log_scores = np.log(shifted_scores)
    
    # Normalize to 1-100 range
    min_log = log_scores.min()
    max_log = log_scores.max()
    
    if max_log > min_log:
        normalized_scores = 1 + 99 * (log_scores - min_log) / (max_log - min_log)
    else:
        normalized_scores = log_scores * 50  # fallback
    
    # Apply 1.333x multiplier
    normalized_scores = normalized_scores * 1.333
    
    return normalized_scores

def assign_grades(normalized_scores: pd.Series) -> pd.Series:
    """
    Assign letter grades based on normalized 1-100 scores with new boundaries.
    A: 80-100, B: 60-80, C: 40-60, D: 20-40, F: 0-20
    """
    grades = pd.Series(index=normalized_scores.index, dtype='object')
    
    grades[normalized_scores >= GRADE_BOUNDARIES["A"]] = "A"
    grades[(normalized_scores >= GRADE_BOUNDARIES["B"]) & (normalized_scores < GRADE_BOUNDARIES["A"])] = "B"
    grades[(normalized_scores >= GRADE_BOUNDARIES["C"]) & (normalized_scores < GRADE_BOUNDARIES["B"])] = "C"
    grades[(normalized_scores >= GRADE_BOUNDARIES["D"]) & (normalized_scores < GRADE_BOUNDARIES["C"])] = "D"
    grades[normalized_scores < GRADE_BOUNDARIES["D"]] = "F"
    
    return grades

def process_dataset_scores(df: pd.DataFrame, score_column: str = "score") -> Tuple[pd.Series, pd.Series]:
    """
    Apply normalization and grading to a dataset.
    Returns (normalized_scores, grades)
    """
    if score_column not in df.columns:
        return pd.Series([], dtype=float), pd.Series([], dtype='object')
    
    original_scores = df[score_column]
    normalized_scores = apply_box_cox_normalization(original_scores)
    grades = assign_grades(normalized_scores)
    
    return normalized_scores, grades

def get_grade_distribution(grades: pd.Series) -> Dict[str, int]:
    """Get grade distribution counts."""
    return grades.value_counts().to_dict()

def get_score_statistics(normalized_scores: pd.Series) -> Dict[str, float]:
    """Get statistics for normalized scores."""
    if len(normalized_scores) == 0:
        return {
            "mean": 0,
            "median": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "count": 0
        }
    
    return {
        "mean": float(normalized_scores.mean()),
        "median": float(normalized_scores.median()),
        "std": float(normalized_scores.std()),
        "min": float(normalized_scores.min()),
        "max": float(normalized_scores.max()),
        "count": int(len(normalized_scores))
    }

if __name__ == "__main__":
    # Test the scoring system
    test_scores = pd.Series([10, 25, 50, 75, 90, 15, 35, 65, 85, 5])
    normalized, grades = process_dataset_scores(pd.DataFrame({"score": test_scores}))
    
    print("Original scores:", test_scores.tolist())
    print("Normalized scores (1.333x):", normalized.round(1).tolist())
    print("Grades:", grades.tolist())
    print("Grade distribution:", get_grade_distribution(grades))
    print("Score statistics:", get_score_statistics(normalized))