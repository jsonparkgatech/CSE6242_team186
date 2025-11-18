"""Minimal fast substitute recommendation using category-based heuristic (no external deps)."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "DATA" / "processed"

# Core nutrient features for fast similarity
FAST_FEATURES = [
    "protein_g_per100g",
    "total_fat_g_per100g", 
    "carbs_g_per100g",
    "energy_kcal_per100g",
    "sugar_g_per100g"
]

def _min_max_scale(values):
    """Simple min-max scaling without sklearn."""
    values = pd.Series(values)
    min_val = values.min()
    max_val = values.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(values), index=values.index)
    return (values - min_val) / (max_val - min_val)

def _compute_simple_similarity(
    base_food: pd.Series, 
    candidate_foods: pd.DataFrame
) -> pd.Series:
    """Compute simple Euclidean similarity without external dependencies."""
    
    # Handle missing features
    for feature in FAST_FEATURES:
        if feature not in candidate_foods.columns:
            candidate_foods[feature] = 0
    
    # Create similarity scores for each feature
    similarities = []
    
    for _, candidate in candidate_foods.iterrows():
        feature_similarities = []
        
        for feature in FAST_FEATURES:
            base_val = base_food.get(feature, 0) or 0
            cand_val = candidate.get(feature, 0) or 0
            
            # Avoid division by zero
            if base_val == 0 and cand_val == 0:
                feature_sim = 1.0  # Both zero, consider similar
            elif base_val == 0 or cand_val == 0:
                feature_sim = 0.0  # One zero, other non-zero
            else:
                # Relative difference similarity
                diff = abs(base_val - cand_val)
                max_val = max(base_val, cand_val)
                feature_sim = 1.0 - (diff / max_val if max_val > 0 else 1.0)
            
            feature_similarities.append(max(0, feature_sim))
        
        # Average similarity across features
        similarities.append(sum(feature_similarities) / len(feature_similarities))
    
    return pd.Series(similarities, index=candidate_foods.index)

def get_fast_substitutions(
    food_id: int,
    k: int = 5,
    constraints: Optional[Dict] = None,
    foods_df: Optional[pd.DataFrame] = None,
    processed_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fast substitute recommendations using category-based heuristic."""
    
    # Default constraints (more permissive for faster results)
    default_constraints = {
        "min_score_gain": 1,  # Reduced from 5
        "max_kcal_delta": 500,  # Increased from 200
        "sodium_cap": 2000,  # Increased from 1000
        "form_match": False,  # Made optional
        "group_match": True,
    }
    if constraints:
        default_constraints.update(constraints)
    
    # Load data
    if foods_df is None:
        data_path = (processed_dir or DATA_DIR) / "foods_nutrients.parquet"
        foods_df = pd.read_parquet(data_path)
    
    # Get base food
    base_food = foods_df[foods_df["fdc_id"] == food_id]
    
    if base_food.empty:
        return pd.DataFrame(columns=["neighbor_id", "score", "grade", "why"])
    
    base_food = base_food.iloc[0]
    
    # Filter candidates by category and form
    category_mask = foods_df["food_category"] == base_food["food_category"]
    
    if default_constraints.get("form_match", True) and pd.notna(base_food.get("form")):
        category_mask &= foods_df["form"] == base_food["form"]
    
    candidates = foods_df[category_mask].copy()
    
    # If no candidates in exact category, relax constraints
    if len(candidates) <= 1:
        # Only match food category
        category_mask = foods_df["food_category"] == base_food["food_category"]
        candidates = foods_df[category_mask].copy()
    
    # If still no candidates, use broader search
    if len(candidates) <= 1:
        candidates = foods_df.copy()
    
    # Remove the base food itself
    candidates = candidates[candidates["fdc_id"] != food_id]
    
    if candidates.empty:
        return pd.DataFrame(columns=["neighbor_id", "score", "grade", "why"])
    
    # Apply additional constraints
    if default_constraints.get("max_kcal_delta") and pd.notna(base_food.get("kcal_per_serv")):
        kcal_diff = abs(candidates["kcal_per_serv"] - base_food["kcal_per_serv"])
        candidates = candidates[kcal_diff <= default_constraints["max_kcal_delta"]]
    
    if default_constraints.get("sodium_cap"):
        candidates = candidates[candidates["sodium_mg_perserving"] <= default_constraints["sodium_cap"]]
    
    if candidates.empty:
        return pd.DataFrame(columns=["neighbor_id", "score", "grade", "why"])
    
    # Score gain filter
    candidates = candidates.copy()
    candidates["score_gain"] = candidates["score"] - base_food["score"]
    candidates = candidates[candidates["score_gain"] >= default_constraints["min_score_gain"]]
    
    if candidates.empty:
        return pd.DataFrame(columns=["neighbor_id", "score", "grade", "why"])
    
    # Compute simple feature similarities
    similarities = _compute_simple_similarity(base_food, candidates)
    candidates["similarity"] = similarities
    
    # Combined scoring: 70% score improvement + 30% nutritional similarity
    candidates["combined_score"] = (
        0.7 * (candidates["score_gain"] / 100.0) + 
        0.3 * candidates["similarity"]
    )
    
    # Sort by combined score
    candidates = candidates.sort_values("combined_score", ascending=False)
    
    # Build explanation
    def fmt_explanation(row):
        parts = [f"score +{int(round(row['score_gain']))}"]
        if pd.notna(row.get("sodium_mg_perserving")):
            parts.append(f"sodium {int(round(row['sodium_mg_perserving']))}mg")
        if pd.notna(row.get("kcal_per_serv")):
            parts.append(f"kcal {int(round(row['kcal_per_serv']))}")
        return ", ".join(parts)
    
    # Take top k results
    results = candidates.head(k).copy()
    results["why"] = [fmt_explanation(row) for _, row in results.iterrows()]
    results["neighbor_id"] = results["fdc_id"]  # Map fdc_id to neighbor_id
    
    return results[["neighbor_id", "score", "score_gain", "grade", "why"]]

if __name__ == "__main__":
    # Test the fast recommendation system
    data_path = DATA_DIR / "foods_nutrients.parquet"
    if data_path.exists():
        print("Loading foods dataset...")
        df = pd.read_parquet(data_path)
        print(f"Dataset: {len(df):,} foods")
        
        # Test with a sample food
        sample_foods = df[df["score"] > 50].head(5)
        if len(sample_foods) > 0:
            test_food_id = sample_foods.iloc[0]["fdc_id"]
            print(f"\nTesting with food ID: {test_food_id}")
            
            results = get_fast_substitutions(test_food_id, k=3)
            print(f"Found {len(results)} substitutes:")
            for _, row in results.iterrows():
                print(f"  - {row['neighbor_id']}: score {row['score']:.1f}, {row['why']}")
        else:
            print("No suitable test foods found.")