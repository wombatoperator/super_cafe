#!/usr/bin/env python3
"""
Unit test to verify leakage fixes work correctly
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from caafe.critic import Critic


def test_critic_no_leakage():
    """Test that the Critic doesn't have data leakage between folds."""
    print("Testing Critic for data leakage...")
    
    # Create a simple dataset
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(200, 10), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(np.random.randint(0, 2, 200), name="is_clicked")
    
    # Create critic with LogisticRegression
    c = Critic(folds=3, repeats=2)
    
    # Score with hold-out
    hold_auc = c.score(X, y)
    
    # Get CV AUC (private access for testing)
    dev_auc = np.mean(c.fold_scores)
    
    print(f"Hold-out AUC: {hold_auc:.4f}")
    print(f"CV AUC: {dev_auc:.4f}")
    print(f"Difference: {abs(hold_auc - dev_auc):.4f}")
    
    # The difference should be reasonable for small datasets
    # Large systematic differences (> 0.1) indicate leakage
    # Some variance is expected due to different data splits
    difference = abs(hold_auc - dev_auc)
    
    if difference < 0.1:
        print("✅ PASS: Difference within acceptable range (no systematic leakage)")
        return True
    else:
        print("❌ FAIL: Large systematic difference suggests leakage")
        return False


def test_reproducibility():
    """Test that LogisticRegression gives reproducible results."""
    print("\nTesting reproducibility...")
    
    # Create same dataset
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100), name="target")
    
    # Score twice with same data
    c1 = Critic(folds=3, repeats=1, seed=42)
    c2 = Critic(folds=3, repeats=1, seed=42)
    
    auc1 = c1.score(X, y)
    auc2 = c2.score(X, y)
    
    print(f"First run AUC: {auc1:.6f}")
    print(f"Second run AUC: {auc2:.6f}")
    print(f"Difference: {abs(auc1 - auc2):.8f}")
    
    # Should be exactly the same with LogisticRegression
    if abs(auc1 - auc2) < 1e-6:
        print("✅ PASS: Results are reproducible")
        return True
    else:
        print("❌ FAIL: Results are not reproducible")
        return False


def test_no_model_contamination():
    """Test that models are properly cloned and don't contaminate each other."""
    print("\nTesting model contamination...")
    
    # Create two different datasets
    np.random.seed(42)
    X1 = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y1 = pd.Series(np.random.randint(0, 2, 100), name="target")
    
    np.random.seed(123)  # Different seed
    X2 = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y2 = pd.Series(np.random.randint(0, 2, 100), name="target")
    
    # Score both datasets with same critic instance
    c = Critic(folds=3, repeats=1, seed=42)
    
    auc1 = c.score(X1, y1)
    auc2 = c.score(X2, y2)
    
    # Score first dataset again - should be the same as first time
    auc1_again = c.score(X1, y1)
    
    print(f"Dataset 1 (first): {auc1:.6f}")
    print(f"Dataset 2: {auc2:.6f}")
    print(f"Dataset 1 (again): {auc1_again:.6f}")
    print(f"Difference (should be 0): {abs(auc1 - auc1_again):.8f}")
    
    if abs(auc1 - auc1_again) < 1e-6:
        print("✅ PASS: No model contamination between datasets")
        return True
    else:
        print("❌ FAIL: Model contamination detected")
        return False


if __name__ == "__main__":
    print("🔬 Testing CAAFE Leakage Fixes")
    print("=" * 50)
    
    results = []
    results.append(test_critic_no_leakage())
    results.append(test_reproducibility())
    results.append(test_no_model_contamination())
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 ALL TESTS PASSED! Leakage fixes are working correctly.")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED! Review the implementation.")
        sys.exit(1)