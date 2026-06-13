#!/usr/bin/env python3
"""
Quick test of frame_similarity_visualizer imports and basic functionality.
This doesn't run the full extraction, just validates the code structure.
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        # Test embedding_analysis imports
        from embedding_analysis.io_cache import load_cache_dir
        from embedding_analysis.io_ground_truth import list_stems_in_ground_truth, load_ground_truth_column
        from embedding_analysis.pipeline import get_carry_intervals, _video_frame_count
        from embedding_analysis.segments import build_segments, middle_index_sorted
        from embedding_analysis.geometry import l2_normalize_rows
        print("  ✓ embedding_analysis imports OK")
    except Exception as e:
        print(f"  ✗ embedding_analysis imports FAILED: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✓ numpy OK")
    except Exception as e:
        print(f"  ✗ numpy FAILED: {e}")
        return False
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        print("  ✓ PIL OK")
    except Exception as e:
        print(f"  ✗ PIL FAILED: {e}")
        return False
    
    try:
        import cv2
        print("  ✓ opencv (cv2) OK")
    except Exception as e:
        print(f"  ✗ opencv FAILED: {e}")
        return False
    
    return True


def test_geometry():
    """Test basic geometry functions."""
    print("\nTesting geometry functions...")
    
    try:
        from embedding_analysis.geometry import l2_normalize_rows, cosine_similarity_matrix
        import numpy as np
        
        # Test L2 normalization
        vec = np.array([[3.0, 4.0]])
        norm = l2_normalize_rows(vec)
        expected_norm = 1.0
        actual_norm = float(np.linalg.norm(norm[0]))
        assert abs(actual_norm - expected_norm) < 1e-6, f"Expected norm {expected_norm}, got {actual_norm}"
        print("  ✓ L2 normalization works")
        
        # Test cosine similarity
        vecs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        cos_mat = cosine_similarity_matrix(vecs)
        assert cos_mat.shape == (3, 3), f"Expected shape (3, 3), got {cos_mat.shape}"
        assert abs(cos_mat[0, 0] - 1.0) < 1e-6, "Diagonal should be 1.0"
        assert abs(cos_mat[0, 1]) < 1e-6, "Orthogonal vectors should have cos=0"
        print("  ✓ Cosine similarity matrix works")
        
        return True
    except Exception as e:
        print(f"  ✗ Geometry tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_middle_index():
    """Test middle index calculation."""
    print("\nTesting middle index calculation...")
    
    try:
        from embedding_analysis.segments import middle_index_sorted
        
        assert middle_index_sorted([1, 2, 3, 4, 5]) == 2, "Middle of 5 items should be index 2"
        assert middle_index_sorted([1, 2, 3, 4]) == 2, "Middle of 4 items should be index 2"
        assert middle_index_sorted([1]) == 0, "Middle of 1 item should be index 0"
        assert middle_index_sorted([]) == -1, "Middle of empty list should be -1"
        print("  ✓ Middle index calculation works")
        
        return True
    except Exception as e:
        print(f"  ✗ Middle index tests FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Frame Similarity Visualizer - Component Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Geometry", test_geometry()))
    results.append(("Middle Index", test_middle_index()))
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! The script should work correctly.")
        print("\nTo run the full frame extraction:")
        print("  ./run_frame_similarity.sh 20 50")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
