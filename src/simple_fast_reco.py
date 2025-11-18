"""Simple fast substitute recommendation using category-based heuristic (no FAISS required)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.preprocessing import MinMaxScaler

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

def _get_eligible_foods(df: pd.DataFrame, min_features: int = 3) -> pd.DataFrame:
    """Filter foods that have sufficient nutrient data."""
    has_serving = df["serving_gram_weight"].notna()
    nutrient_data = df[FAST_FEATURES].notna().sum(axis=1) >= min_features
    return df[has_serving & nutrient_data].copy()

def _compute_feature_similarity(
    base_food: pd.Series, 
    candidate_foods: pd.DataFrame
) -> np.ndarray:
    """Compute simple feature-based similarity scores."""
    
    # Extract features
    base_features = base_food[FAST_FEATURES].fillna(0).values
    
    # Handle case where candidate doesn't have the features
    if FAST_FEATURES[0] not in candidate_foods.columns:
        return np.zeros(len(candidate_foods))
    
    candidate_features = candidate_foods[FAST_FEATURES].fillna(0).values
    
    # MinMax scale both
    scaler = MinMaxScaler()
    
    # Fit on combined data to ensure consistent scaling
    all_features = np.vstack([base_features.reshape(1, -1), candidate_features])
    scaler.fit(all_features)
    
    base_scaled = scaler.transform(base_features.reshape(1, -1)).flatten()
    candidate_scaled = scaler.transform(candidate_features)
    
    # Compute L2 distances and convert to similarities
    distances = np.sqrt(np.sum((candidate_scaled - base_scaled) ** 2, axis=1))
    
    # Convert distance to similarity (higher similarity for lower distance)
    # Use exponential decay to get similarity scores between 0 and 1
    max_distance = np.sqrt(len(FAST_FEATURES))  # Maximum possible L2 distance
    similarities = np.exp(-distances / (max_distance / 2))
    
    return similarities

def get_fast_substitutions(
    food_id: int,
    k: int = 5,
    constraints: Optional[Dict] = None,
    foods_df: Optional[pd.DataFrame] = None,
    processed_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fast substitute recommendations using category-based heuristic."""
    
    # Default constraints
    default_constraints = {
        "min_score_gain": 5,
        "max_kcal_delta": 200,
        "sodium_cap": 1000,
        "form_match": True,
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
    
    # Compute feature similarities
    similarities = _compute_feature_similarity(base_food, candidates)
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
    
    return results[["neighbor_id", "score", "score_gain", "grade", "why"]]

def preload_category_cache(foods_df: pd.DataFrame) -> Dict:
    """Pre-compute category-based caches for faster lookup."""
    cache = {}
    
    # Group foods by category
    categories = foods_df["food_category"].dropna().unique()
    
    for category in categories:
        category_foods = foods_df[foods_df["food_category"] == category]
        
        # Pre-sort by score for faster access
        category_foods = category_foods.sort_values("score", ascending=False)
        
        # Also group by form within category
        forms = category_foods["form"].dropna().unique()
        cache[category] = {
            "all": category_foods,
            "by_form": {form: category_foods[category_foods["form"] == form] for form in forms}
        }
    
    return cache

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