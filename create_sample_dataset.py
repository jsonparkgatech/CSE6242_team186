#!/usr/bin/env python3
"""
Create a 50k random sample from the full dataset for faster interactive features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_dataset():
    """Create a 50k random sample from the full dataset."""
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Load the full dataset
    print("Loading full dataset...")
    data_path = Path("DATA/processed/foods_nutrients.parquet")
    
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        return False
    
    full_df = pd.read_parquet(data_path)
    print(f"Full dataset size: {len(full_df):,} records")
    
    # Filter out foods with score = 0 (but keep NaN scores as they may be in the process of being calculated)
    if 'score' in full_df.columns:
        non_zero_df = full_df[full_df['score'] != 0].copy()
        print(f"Foods with non-zero scores: {len(non_zero_df):,} out of {len(full_df):,}")
    else:
        print("Warning: 'score' column not found, using all foods")
        non_zero_df = full_df.copy()
    
    # Create 50k random sample from non-zero scored foods
    sample_size = min(50000, len(non_zero_df))
    print(f"Creating sample of {sample_size:,} records from non-zero scored foods...")
    
    sample_df = non_zero_df.sample(n=sample_size, random_state=42)
    
    # Save the sample
    sample_path = Path("DATA/processed/foods_sample_50k.parquet")
    sample_df.to_parquet(sample_path, index=False)
    print(f"Sample saved to: {sample_path}")
    
    # Show sample statistics
    print(f"\nSample statistics:")
    print(f"- Records: {len(sample_df):,}")
    print(f"- Columns: {len(sample_df.columns)}")
    print(f"- Unique food categories: {sample_df['food_category'].nunique() if 'food_category' in sample_df.columns else 'N/A'}")
    print(f"- Source: Random sample from {len(non_zero_df):,} non-zero scored foods")
    
    if 'score' in sample_df.columns:
        print(f"- Score range: {sample_df['score'].min():.1f} to {sample_df['score'].max():.1f}")
        print(f"- Average score (excludes 0): {sample_df['score'].mean():.1f}")
        # Show percentage of foods with scores in the original dataset
        total_with_scores = len(full_df[full_df['score'].notna()])
        non_zero_count = len(full_df[full_df['score'] != 0])
        print(f"- Original dataset: {non_zero_count:,} non-zero scored foods out of {total_with_scores:,} with scores ({non_zero_count/total_with_scores*100:.1f}%)")
    
    return True

if __name__ == "__main__":
    create_sample_dataset()