"""Fast substitute recommendation using FAISS and category-based heuristic."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
CFGS = ROOT / "configs"
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

def _build_fast_index(foods_df: pd.DataFrame) -> Tuple[faiss.Index, pd.DataFrame]:
    """Build FAISS index with simplified features."""
    print(f"Building FAISS index with {len(foods_df):,} foods...")
    
    # Extract and scale features
    feature_matrix = foods_df[FAST_FEATURES].fillna(0).values
    
    # MinMax scaling (faster than robust scaling)
    scaler = MinMaxScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    
    # Convert to float32 for FAISS
    feature_matrix = feature_matrix.astype(np.float32)
    
    # Build FAISS index (L2 distance for MinMax scaled data)
    dimension = feature_matrix.shape[1]
    nlist = min(100, len(foods_df) // 1000)  # Number of clusters
    
    if len(foods_df) > 10000:
        # Use IVF for large datasets
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        index.train(feature_matrix)
    else:
        # Use flat index for smaller datasets
        index = faiss.IndexFlatL2(dimension)
    
    index.add(feature_matrix)
    print(f"FAISS index built: {index.ntotal} vectors, {index.d} dimensions")
    
    return index, foods_df.reset_index(drop=True)

def _category_based_search(
    index: faiss.Index, 
    foods_df: pd.DataFrame, 
    query_fdc_id: int, 
    k: int = 10
) -> pd.DataFrame:
    """Fast category-based heuristic search."""
    try:
        base_food_idx = foods_df[foods_df["fdc_id"] == query_fdc_id].index[0]
    except IndexError:
        return pd.DataFrame()
    
    base_food = foods_df.iloc[base_food_idx]
    
    # Filter by same category and form
    category_mask = foods_df["food_category"] == base_food["food_category"]
    if pd.notna(base_food.get("form")):
        category_mask &= foods_df["form"] == base_food["form"]
    
    category_candidates = foods_df[category_mask].copy()
    
    if len(category_candidates) == 0:
        # Fallback: only category match
        category_mask = foods_df["food_category"] == base_food["food_category"]
        category_candidates = foods_df[category_mask].copy()
    
    if len(category_candidates) <= 1:
        # Fallback: expand to broader search
        category_candidates = foods_df.copy()
    
    # Score-based ranking with minimum improvement
    min_score_gain = 5
    if base_food["score"] > 0:
        better_foods = category_candidates[category_candidates["score"] > base_food["score"] + min_score_gain]
        if len(better_foods) >= k:
            return better_foods.nlargest(k, "score")
        elif len(better_foods) > 0:
            return better_foods
    
    # If no better foods in category, get top scored foods
    return category_candidates.nlargest(k, "score")

def get_fast_substitutions(
    food_id: int,
    k: int = 5,
    constraints: Optional[Dict] = None,
    foods_df: Optional[pd.DataFrame] = None,
    processed_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fast substitute recommendations using FAISS + category heuristic."""
    
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
    
    # Get eligible foods
    eligible_foods = _get_eligible_foods(foods_df)
    base_food = foods_df[foods_df["fdc_id"] == food_id]
    
    if base_food.empty:
        return pd.DataFrame(columns=["neighbor_id", "score", "grade", "why"])
    
    base_food = base_food.iloc[0]
    
    # Fast category-based search
    candidates = _category_based_search(None, eligible_foods, food_id, k=50)
    
    if candidates.empty:
        return pd.DataFrame(columns=["neighbor_id", "score", "grade", "why"])
    
    # Apply constraints
    if default_constraints.get("max_kcal_delta") and pd.notna(base_food.get("kcal_per_serv")):
        kcal_delta = abs(candidates["kcal_per_serv"] - base_food["kcal_per_serv"])
        candidates = candidates[kcal_delta <= default_constraints["max_kcal_delta"]]
    
    if default_constraints.get("sodium_cap"):
        candidates = candidates[candidates["sodium_mg_perserving"] <= default_constraints["sodium_cap"]]
    
    # Score gain filter
    candidates["score_gain"] = candidates["score"] - base_food["score"]
    candidates = candidates[candidates["score_gain"] >= default_constraints["min_score_gain"]]
    
    # Sort by score gain (highest improvement first)
    candidates = candidates.sort_values("score_gain", ascending=False)
    
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

def build_fast_index(
    foods_df: pd.DataFrame, 
    output_path: Optional[Path] = None
) -> Tuple[faiss.Index, pd.DataFrame]:
    """Build and optionally save fast FAISS index."""
    
    eligible_foods = _get_eligible_foods(foods_df)
    index, foods_with_index = _build_fast_index(eligible_foods)
    
    # Save metadata
    if output_path:
        metadata = {
            "features": FAST_FEATURES,
            "food_count": len(eligible_foods),
            "feature_dimension": index.d,
            "index_type": type(index).__name__
        }
        
        import json
        with open(output_path / "fast_index_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    return index, foods_with_index

if __name__ == "__main__":
    # Test the fast recommendation system
    data_path = DATA_DIR / "foods_nutrients.parquet"
    if data_path.exists():
        print("Loading foods dataset...")
        df = pd.read_parquet(data_path)
        print(f"Dataset: {len(df):,} foods")
        
        # Build index
        index, foods = build_fast_index(df, output_path=DATA_DIR)
        
        # Test with first food
        if len(foods) > 0:
            test_food_id = foods.iloc[0]["fdc_id"]
            print(f"\nTesting with food ID: {test_food_id}")
            
            results = get_fast_substitutions(test_food_id, k=3)
            print(f"Found {len(results)} substitutes:")
            for _, row in results.iterrows():
                print(f"  - {row['neighbor_id']}: score {row['score']:.1f}, {row['why']}")