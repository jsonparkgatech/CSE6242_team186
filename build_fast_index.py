#!/usr/bin/env python3
"""
Build fast FAISS index for substitute recommendations.
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src"))

from fast_reco import build_fast_index
import pandas as pd

def main():
    print("Building fast FAISS index for substitute recommendations...")
    
    # Check if FAISS is available
    try:
        import faiss
        print(f"✓ FAISS version: {faiss.__version__}")
    except ImportError:
        print("❌ FAISS not installed. Install with: pip install faiss-cpu")
        return False
    
    # Load data
    data_path = Path("DATA/processed/foods_nutrients.parquet")
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        return False
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Dataset: {len(df):,} foods")
    
    # Build fast index
    print("Building fast FAISS index...")
    try:
        index, foods = build_fast_index(df, output_path=Path("DATA/processed"))
        print(f"✓ Index built successfully!")
        print(f"  - {len(foods):,} eligible foods")
        print(f"  - {index.d} dimensional features")
        print(f"  - Index type: {type(index).__name__}")
        
        # Save index to disk
        import pickle
        index_path = Path("DATA/processed/fast_index.pkl")
        foods_path = Path("DATA/processed/fast_foods.pkl")
        
        with open(index_path, "wb") as f:
            pickle.dump(index, f)
        
        with open(foods_path, "wb") as f:
            pickle.dump(foods, f)
            
        print(f"✓ Index saved to: {index_path}")
        print(f"✓ Foods metadata saved to: {foods_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error building index: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Fast index ready for substitute recommendations!")
    else:
        print("\n💥 Failed to build fast index.")
        sys.exit(1)