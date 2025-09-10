#!/usr/bin/env python3
"""Test script for data collection with small sample."""

import subprocess
import sys
from pathlib import Path

def run_test_collection():
    """Run a small test collection."""
    print("Running test collection with small sample...")
    
    # Test with 2 models, 2 datasets, 2 samples each
    cmd = [
        "python", "scripts/collect_all_data.py",
        "--data_root", "storage/datasets",
        "--model_root", "storage/models", 
        "--output_dir", "storage/test_runs",
        "--models", "Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2",
        "--datasets", "gsm8k", "mgsm",
        "--max_samples_per_dataset", "2",
        "--max_new_tokens", "512",
        "--top_p", "1.0",
        "--language", "en",
        "--seed", "42",
        "--verbose"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Test collection completed successfully!")
            
            # Check output structure
            test_output = Path("storage/test_runs")
            if test_output.exists():
                print("\n📁 Output structure:")
                for item in test_output.rglob("*"):
                    if item.is_file():
                        print(f"  {item}")
            
            return True
        else:
            print("❌ Test collection failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test collection timed out!")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

if __name__ == "__main__":
    success = run_test_collection()
    sys.exit(0 if success else 1)
